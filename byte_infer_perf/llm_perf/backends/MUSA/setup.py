from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.MUSA.musa_engine import MUSAEngine
from llm_perf.backends.MUSA.musa_sampler import MUSASampler
from llm_perf.backends.MUSA.musa_scheduler import MUSAScheduler
from llm_perf.utils.logger import logger

def setup_scheduler(
    model_config: Dict[str, Any], 
    pad_token_id, max_batch_size, 
    **kwargs
) -> CoreScheduler:
    # create engine
    engine = MUSAEngine(model_config, pad_token_id)

    # create sampler
    sampler = MUSASampler()

    # create scheduler
    scheduler = MUSAScheduler(
        engine=engine, 
        sampler=sampler, 
        max_batch_size=max_batch_size
    )

    return scheduler