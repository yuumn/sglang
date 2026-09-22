"""LatentSpec V2 worker built on the linear-block DFlash protocol."""

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class LatentSpecWorkerV2(DFlashWorkerV2):
    def __init__(self, *args, **kwargs):
        ps = kwargs.get("ps")
        if ps is None and len(args) >= 3:
            ps = args[2]
        if ps is not None and int(ps.tp_size) != 1:
            raise ValueError("LATENTSPEC draft execution currently requires --tp-size 1.")
        super().__init__(*args, **kwargs)
        required = ("num_latent_tokens", "num_latent_layers", "latent_token_id")
        missing = [name for name in required if not hasattr(self.draft_model, name)]
        if missing:
            raise ValueError(
                "LATENTSPEC requires a Qwen3MySpecModel draft checkpoint; "
                f"missing model attributes: {missing}."
            )

    def alloc_memory_pool(self, *args, **kwargs):
        super().alloc_memory_pool(*args, **kwargs)
        self.draft_model.set_runtime_pools(
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
        )

    def init_cuda_graphs(self):
        # The draft has two different logical widths (latent and proposal), so
        # it intentionally stays eager until a dedicated graph runner exists.
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
