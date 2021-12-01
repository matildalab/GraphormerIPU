python3 srcIPU/entry.py --mini_batch_size 28 --gradient_accumulation 32 --replication_factor 4 --device_iteration 1 --pipeline_splits layers/1 layers/5 layers/8 \
                    --checkpoint 0 --test 1
