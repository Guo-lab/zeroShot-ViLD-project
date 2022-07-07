# zero_shot_object_detection
### Debug in tpu_executor.py the predictor in estimator

Just trapped in the iterator[outputs = next(predictor)]

Try: 
1. Output LOG info in main.py
2. Add Hooks and visual the graph in tensorboard (graph.png and graph, tf_event_local file)
3. Give all kinds of checkpoints with print()
