from lib.dequedict import DequeDict
from lib.heapdict import HeapDict
import numpy as np


class LeCaR:
    ######################
    ## INTERNAL CLASSES ##
    ######################

    # Entry to track the page information
    class LeCaR_Entry:
        def __init__(self, oblock, freq=1, time=0):
            self.oblock = oblock
            self.freq = freq
            self.time = time
            self.evicted_time = None

        # Minimal comparitors needed for HeapDict
        def __lt__(self, other):
            if self.freq == other.freq:
                return self.oblock < other.oblock
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    # kwargs: We're using keyword arguments so that they can be passed down as
    #         needed. Please note that cache_size is a required argument and not
    #         optional like all the kwargs are.
    def __init__(self, cache_size, prefetch_count=0, **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0
        self.prefetch_count = prefetch_count  # 新增：预取页数

        # Cache
        self.cache_size = cache_size
        self.lru = DequeDict()
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size
        self.lru_hist = DequeDict()
        self.lfu_hist = DequeDict()

        # Decision Weights Initilized
        self.initial_weight = 0.5

        # Fixed Learning Rate
        self.learning_rate = 0.45

        # Fixed Discount Rate
        self.discount_rate = 0.005**(1 / self.cache_size)

        # Decision Weights
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

    # True if oblock is in cache (which LRU can represent)
    def __contains__(self, oblock):
        return oblock in self.lru

    def set_prefetch_count(self, count):
        """动态调整预取页数"""
        self.prefetch_count = count

    def cacheFull(self):
        return len(self.lru) == self.cache_size

    # Add Entry to cache with given frequency
    def addToCache(self, oblock, freq):
        x = self.LeCaR_Entry(oblock, freq, self.time)
        self.lru[oblock] = x
        self.lfu[oblock] = x

    # Add Entry to history dictated by policy
    # policy: 0, Add Entry to LRU History
    #         1, Add Entry to LFU History
    def addToHistory(self, x, policy):
        # Use reference to policy_history to reduce redundant code
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            return

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            evicted = self.getLRU(policy_history)
            del policy_history[evicted.oblock]
        policy_history[x.oblock] = x

    # Get the LRU item in the given DequeDict
    def getLRU(self, dequeDict):
        return dequeDict.first()

    def getHeapMin(self):
        return self.lfu.min()

    # Get the random eviction choice based on current weights
    def getChoice(self):
        return 0 if np.random.rand() < self.W[0] else 1

    # Evict an entry
    def evict(self):
        lru = self.getLRU(self.lru)
        lfu = self.getHeapMin()

        evicted = lru
        policy = self.getChoice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
        else:
            evicted = lfu

        del self.lru[evicted.oblock]
        del self.lfu[evicted.oblock]

        evicted.evicted_time = self.time

        self.addToHistory(evicted, policy)

        return evicted.oblock, policy
    
    def getHotPages(self, k=10, alpha=0.7, beta=0.3):
        """
        返回当前缓存中最热的top-k数据页
        参数:
        k: 返回的热门页面数量
        alpha: 频率权重 (默认0.7)
        beta: 新近度权重 (默认0.3)
        
        返回:
        按热度降序排列的页面oblock列表
        """
        if not self.lru or k <= 0:
            return []
        
        current_time = self.time
        pages_with_score = []
        
        # 为缓存中的每个页面计算热度分数
        for entry in self.lru:
            # 频率部分 - 直接使用访问次数
            freq_score = entry.freq
            
            # 新近度部分 - 使用时间衰减因子，越近访问的页面得分越高
            # 使用指数衰减: e^(-λ*(current_time - entry.time))
            # 这里λ是一个衰减系数，可以根据需要调整
            recency_score = np.exp(-0.05 * (current_time - entry.time))
            
            # 综合分数 = α * 频率 + β * 新近度
            heat_score = alpha * freq_score + beta * recency_score
            
            pages_with_score.append((entry.oblock, heat_score, entry.freq, current_time - entry.time))
        
        # 按热度分数降序排序
        pages_with_score.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个页面的oblock
        return [page[0] for page in pages_with_score[:k]]

    # Cache Hit
    def hit(self, oblock):
        x = self.lru[oblock]
        x.time = self.time

        self.lru[oblock] = x

        x.freq += 1
        self.lfu[oblock] = x

    # Adjust the weights based on the given rewards for LRU and LFU
    def adjustWeights(self, rewardLRU, rewardLFU):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    # Cache Miss
    def miss(self, oblock):
        evicted = None

        freq = 1
        if oblock in self.lru_hist:
            entry = self.lru_hist[oblock]
            freq = entry.freq + 1
            del self.lru_hist[oblock]
            reward_lru = -(self.discount_rate
                           **(self.time - entry.evicted_time))
            self.adjustWeights(reward_lru, 0)
        elif oblock in self.lfu_hist:
            entry = self.lfu_hist[oblock]
            freq = entry.freq + 1
            del self.lfu_hist[oblock]
            reward_lfu = -(self.discount_rate
                           **(self.time - entry.evicted_time))
            self.adjustWeights(0, reward_lfu)

        # If the cache is full, evict
        if len(self.lru) == self.cache_size:
            evicted, policy = self.evict()

        self.addToCache(oblock, freq)

        return evicted

    # Process and access request for the given oblock
    def request(self, oblock):
        miss = True
        evicted = None

        self.time += 1

        if oblock in self:
            miss = False
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)
            # 顺序预取
            for i in range(1, self.prefetch_count + 1):
                prefetch_block = oblock + i
                if prefetch_block not in self:
                    self.miss(prefetch_block)

        return miss, evicted

    def preload(self, pages):
        for page in pages:
            self.request(page)

    def snapshot(self):
        return set(self.lru)
