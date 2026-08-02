# evaluation/__init__.py
#
# Eval algorithms (calibration, metrics, AUC) now come from the reference
# reference implementation, reference under classifier_engine/reference/. The
# adapter at evaluation/adapter.py orchestrates the calls. The modules here
# (inference, ruleset_builder, model_cache, realtime) are the LOCAL pieces
# that bridge our DB / FastAPI surface to that reference core.
