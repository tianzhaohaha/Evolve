save_path=$DATA_ROOT/searchR1

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000 \

# CUDA_VISIBLE_DEVICES=2,3,4 nohup python -u \
#   examples/search/retriever/retrieval_server.py \
#   --index_path "$DATA_ROOT/searchR1/e5_Flat.index" \
#   --corpus_path "$DATA_ROOT/searchR1/wiki-18.jsonl" \
#   --topk 3 \
#   --retriever_name e5 \
#   --retriever_model intfloat/e5-base-v2 \
#   --faiss_gpu \
#   --port 8001 \
#   > retrieval_server.log 2>&1 &