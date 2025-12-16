from worker import Worker

import numpy as np
import heapq
from dataclasses import dataclass
import torch

@dataclass
class WorkerState:
    worker_id: int
    finish_time: float
    
    def __lt__(self, other):
        return self.finish_time < other.finish_time

class BaseGD:
    def __init__(self, initial_x, data, time_distributions, loss_fn, gradient_fns, learning_rate=0.1, compression_flag = 'none', compression_size = 100):
        """Base class for gradient descent implementations"""
        assert len(gradient_fns) == len(time_distributions), f"Number of gradient functions ({len(gradient_fns)}) \
              must match number of time distributions ({len(time_distributions)})"
        
        self.workers = [
            Worker(loss_fn, gradient_fns[i], time_distributions[i], compression_flag=compression_flag, k=compression_size)
            for i in range(len(gradient_fns))
        ]
        self.current_x = initial_x
        self.learning_rate = learning_rate
        self.loss_history = []
        self.grad_norm_history = []
        self.computation_times = []
        self.current_time = 0
        self.x_history = [initial_x]
        self.data = data
        self.loss_fn = loss_fn

    def _update_loss_history(self):
        """Update loss history with current total loss"""
        total_loss = self.loss_fn(self.data, self.current_x)
        self.loss_history.append(total_loss)

    @staticmethod
    def _as_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _record_grad_norm(self, grad):
        g = self._as_numpy(grad)
        self.grad_norm_history.append(float(np.linalg.norm(g)))

class MinibatchSGD(BaseGD):
    def run_steps(self, num_steps):
        for _ in range(num_steps):
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def run_until_time(self, max_time):
        """Run until simulated time exceeds max_time."""
        while self.current_time < max_time:
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def _step(self):
        """Execute one step of MinibatchSGD."""
        gradients_and_times = [
            worker.compute_gradient(self.data, self.current_x)
            for worker in self.workers
        ]
        
        gradients, times = zip(*gradients_and_times)
        gradients = [self._as_numpy(grad) for grad in gradients]
        avg_gradient = np.mean(gradients, axis=0)
        self._record_grad_norm(avg_gradient)
        
        self.current_x = self.current_x - self.learning_rate * avg_gradient
        self.current_time += np.max(times)
        self.computation_times.append(self.current_time)
        self._update_loss_history()
        self.x_history.append(self.current_x)

class AsynchronousGD(BaseGD):
    def _init_heap(self):
        """Initialize the worker heap."""
        self._heap = []
        for i, worker in enumerate(self.workers):
            gradient, time = worker.compute_gradient(self.data, self.current_x)
            heapq.heappush(self._heap, WorkerState(i, time + self.current_time))
    
    def run_steps(self, num_steps):        
        self._init_heap()
        for _ in range(num_steps):
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def run_until_time(self, max_time):
        """Run until simulated time exceeds max_time."""
        self._init_heap()
        while self.current_time < max_time:
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def _step(self):
        """Execute one step of AsynchronousGD."""
        worker_state = heapq.heappop(self._heap)
        worker = self.workers[worker_state.worker_id]
        self.current_time = worker_state.finish_time
        
        used_gradient = self._as_numpy(worker.gradient)
        self._record_grad_norm(used_gradient)
        self.current_x = self.current_x - self.learning_rate * used_gradient
        
        gradient, compute_time = worker.compute_gradient(self.data, self.current_x)
        heapq.heappush(self._heap, WorkerState(worker_state.worker_id, self.current_time + compute_time))
        
        self.computation_times.append(self.current_time)
        self._update_loss_history()
        self.x_history.append(self.current_x)

class RennalaSGD(BaseGD):
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def run_steps(self, num_steps):
        for _ in range(num_steps):
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def run_until_time(self, max_time):
        """Run until simulated time exceeds max_time."""
        while self.current_time < max_time:
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history
    
    def _step(self):
        """Execute one step of RennalaSGD."""
        heap = []
        for i, worker in enumerate(self.workers):
            gradient, time = worker.compute_gradient(self.data, self.current_x)
            heapq.heappush(heap, WorkerState(i, time + self.current_time))
        current_gradient = np.zeros_like(self.current_x)
        
        for s in range(self.batch_size):
            state = heapq.heappop(heap)
            worker = self.workers[state.worker_id]
            
            self.current_time = state.finish_time
            
            current_gradient += self._as_numpy(worker.gradient)
            
            _, compute_time = worker.compute_gradient(self.data, self.current_x)
            state.finish_time = self.current_time + compute_time
            heapq.heappush(heap, state)
        
        used_gradient = current_gradient / self.batch_size
        self._record_grad_norm(used_gradient)
        self.current_x = self.current_x - self.learning_rate * used_gradient
        self.computation_times.append(self.current_time)
        self._update_loss_history()
        self.x_history.append(self.current_x)

class AsynchronousNAG(BaseGD):
    """
    Asynchronous Nesterov Accelerated Gradient (NAG).

    Implementation note: each worker computes a gradient at a *lookahead point*
    x + mu*v, but due to asynchrony that gradient can be stale by the time it is applied.

    Because stale gradients compound through velocity, the default momentum is
    conservative (0.5) compared to synchronous NAG (typically 0.9).
    """
    def __init__(
        self,
        initial_x,
        data,
        time_distributions,
        loss_fn,
        gradient_fns,
        learning_rate=0.1,
        momentum=0.5,  # conservative default for async (sync NAG uses ~0.9)
        compression_flag='none',
        compression_size=100,
    ):
        super().__init__(
            initial_x=initial_x,
            data=data,
            time_distributions=time_distributions,
            loss_fn=loss_fn,
            gradient_fns=gradient_fns,
            learning_rate=learning_rate,
            compression_flag=compression_flag,
            compression_size=compression_size,
        )
        self.momentum = momentum
        self.velocity = np.zeros_like(self.current_x)

    def _lookahead_x(self):
        return self.current_x + self.momentum * self.velocity

    def _init_heap(self):
        """Initialize the worker heap with gradients at the current lookahead point."""
        self._heap = []
        lookahead = self._lookahead_x()
        for i, worker in enumerate(self.workers):
            _, time = worker.compute_gradient(self.data, lookahead)
            heapq.heappush(self._heap, WorkerState(i, time + self.current_time))

    def run_steps(self, num_steps):
        self._init_heap()
        for _ in range(num_steps):
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history

    def run_until_time(self, max_time):
        """Run until simulated time exceeds max_time."""
        self._init_heap()
        while self.current_time < max_time:
            self._step()
        return self.current_x, self.loss_history, self.computation_times, self.x_history

    def _step(self):
        worker_state = heapq.heappop(self._heap)
        worker = self.workers[worker_state.worker_id]
        self.current_time = worker_state.finish_time

        # Apply the (possibly stale) gradient computed at the lookahead point.
        g = self._as_numpy(worker.gradient)
        self._record_grad_norm(g)

        self.velocity = self.momentum * self.velocity - self.learning_rate * g
        self.current_x = self.current_x + self.velocity

        # Schedule this worker to compute the next gradient at the updated lookahead.
        lookahead = self._lookahead_x()
        _, compute_time = worker.compute_gradient(self.data, lookahead)
        heapq.heappush(self._heap, WorkerState(worker_state.worker_id, self.current_time + compute_time))

        self.computation_times.append(self.current_time)
        self._update_loss_history()
        self.x_history.append(self.current_x)