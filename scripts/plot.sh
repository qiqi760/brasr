  python ./python_scripts/batch_evaluate_and_plot.py 
      --checkpoint_dir  ./exp/20260501-104455-libri-960-d2v-large-bert-multi-proj512-bs2-freeze999-detach
      --task contrastive-learning 
      --dataset libri-960-dev-bias-new 
      --per_sample_bias_dir ./data/contrastive-learning/per_sample_bias 
      --model_config ./configs/model_config.yaml 
      --output_dir results/batch_eval 
      --device cuda