from __future__ import annotations
from ..core import abc
from .comm import Comm
from typing import Any, Iterable, List
from collections import OrderedDict

class Work(abc.HasDictRepr):
    
    repr_attr_keys = ('content', 'reply')
    
    def __init__(self, content: Any, **kw) -> None:
        
        super().__init__(**kw)

        self.content = content
        self.reply = None


class Pool(abc.HasDictRepr):
    '''
    Usual Usage
    -----------
    pool = Pool(comm)
    
    if pool.is_leader:
        for i in range(100):
            pool.assign_work(i)
        pool.wait_all()                 # wait for all results/replies
        replies = pool.pop_replies()

        for i in range(100):
            pool.assign_work(i+100)
        pool.join()                     # wait() + signal endings to workers
        replies.extend(pool.pop_replies()) 
    else:
        for work in pool.works():
            work.reply = fn(work.content)
    '''
    
    tag             = 0
    signal_work     = 0
    signal_done     = 1
    
    repr_attr_keys = ('comm', 'leader_rank')
    
    def __init__(self, comm: Comm = None, leader_rank: int = 0, 
                 gather_reply: bool = True, **kw) -> None:
        
        super().__init__(**kw)
        if comm is None:
            comm = Comm.world()
        assert comm.size >= 2
        assert leader_rank >= 0 and leader_rank < comm.size
        comm = comm.dup()
        
        if comm.rank == leader_rank:
            workers = OrderedDict([
                (rank, comm.connect(rank, tag=Pool.tag))
                for rank in range(comm.size) if rank != leader_rank
            ])
            free_set = {rank for rank in workers.keys()}
            
            self.workers, self.free_set = workers, free_set
            self.n_workers = comm.size - 1
            self.gather_reply = gather_reply
            self.replies = []
        else:
            self.leader = comm.connect(leader_rank, tag=Pool.tag)
                
        self.comm = comm
        self.leader_rank = leader_rank
        
    @property
    def is_leader(self) -> bool:
        return self.comm.rank == self.leader_rank
        
    @property
    def is_worker(self) -> bool:
        return not self.is_leader
        
    def __enter__(self) -> Pool:
        return self
    
    def __exit__(self, *exc) -> None:
        if self.is_leader:
            self.join()
        
    def pop_replies(self) -> List[Any]:
        replies, self.replies = self.replies, []
        return replies
        
    @property
    def works(self) -> Iterable[Work]:
        '''
        Cannot use `break` to jump out the iterations.
        '''
        assert self.is_worker
        rank, leader = self.comm.rank, self.leader
        signal_work, signal_done = Pool.signal_work, Pool.signal_done
        
        while True:
            msg: dict = leader.recv()
            
            signal = msg['signal']
            if signal == signal_done:
                return
            assert signal == signal_work
            
            work = Work(msg['content'])
            yield work
            msg = {
                'rank': rank, 'reply': work.reply
            }
            leader.send(msg)
            
    def assign_work(self, content: Any) -> None:
        assert self.is_leader
        workers, free_set = self.workers, self.free_set
        try:
            rank = free_set.pop()
        except KeyError:
            rank = self.__wait_reply()
            
        workers[rank].send({
            'signal': Pool.signal_work, 
            'content': content
        })
        
    def wait_all(self) -> None:
        assert self.is_leader
        free_set = self.free_set
        n_workers, n_free = self.n_workers, len(free_set)
        for _ in range(n_free, n_workers):
            free_set.add(self.__wait_reply())
            
    def join(self) -> None:
        assert self.is_leader
        
        self.wait_all()
        
        for worker in self.workers.values():
            worker.send({'signal': Pool.signal_done})
            
    def __wait_reply(self) -> int:
        msg = self.comm.recv(tag=Pool.tag)
        if self.gather_reply:
            self.replies.append(msg['reply'])
        return msg['rank']