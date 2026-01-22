# Checks

lint:
		pylint easyfsl scripts

test:
		pytest easyfsl

isort:
		isort easyfsl scripts

isort-check:
		isort easyfsl scripts --check

black:
		black easyfsl scripts

black-check:
		black easyfsl scripts --check

mypy:
		mypy easyfsl scripts

# Install

dev-install:
		pip install -r dev_requirements.txt

# Download data

download-cub:
	mkdir -p data/CUB
	wget -O data/CUB/images.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
	tar --exclude='._*' --strip-components=1 -zxvf data/CUB/images.tgz -C data/CUB/

# Benchmarks

BATCH_SIZE=1024
NUM_WORKERS=12
MODEL_CHECKPOINTS_DIR=data/models
DEVICE=cuda

extract-mini-imagenet-features-with-resnet12:
	python -m scripts.predict_embeddings \
		feat_resnet12 \
		${MODEL_CHECKPOINTS_DIR}/feat_resnet12_mini_imagenet.pth \
		mini_imagenet \
		--device=${DEVICE} \
		--num-workers=${NUM_WORKERS} \
		--batch-size=${BATCH_SIZE}

extract-features-with-resnet12-trained-on-tiered-imagenet:
	for target_dataset in cub tiered_imagenet fungi; do \
		python -m scripts.predict_embeddings \
			feat_resnet12 \
			${MODEL_CHECKPOINTS_DIR}/feat_resnet12_tiered_imagenet.pth \
			$${target_dataset} \
			--device=${DEVICE} \
			--num-workers=${NUM_WORKERS} \
			--batch-size=${BATCH_SIZE}; \
	done; \

extract-all-features-with-resnet12:
	make extract-mini-imagenet-features-with-resnet12 ; \
	make extract-features-with-resnet12-trained-on-tiered-imagenet ; \

benchmark-mini-imagenet:
	for n_shot in 1 5; do \
		for method in bd_cspn prototypical_networks simple_shot tim finetune laplacian_shot pt_map transductive_finetuning; do \
			python -m scripts.benchmark_methods \
				$${method} \
				data/features/mini_imagenet/test/feat_resnet12_mini_imagenet.parquet.gzip \
				--config="default" \
				--n-shot=$${n_shot} \
				--device=${DEVICE} \
				--num-workers=${NUM_WORKERS}; \
		done; \
		python -m scripts.benchmark_methods \
			feat \
			data/features/mini_imagenet/test/feat_resnet12_mini_imagenet.parquet.gzip \
			--config="resnet12_mini_imagenet" \
			--n-shot=$${n_shot} \
			--device=${DEVICE} \
			--num-workers=${NUM_WORKERS}; \
	done

benchmark-tiered-imagenet:
	for n_shot in 1 5; do \
		for method in bd_cspn prototypical_networks simple_shot tim finetune laplacian_shot pt_map transductive_finetuning; do \
			python -m scripts.benchmark_methods \
				$${method} \
				data/features/tiered_imagenet/test/feat_resnet12_tiered_imagenet.parquet.gzip \
				--config="default" \
				--n-shot=$${n_shot} \
				--device=${DEVICE} \
				--num-workers=${NUM_WORKERS}; \
		done; \
		python -m scripts.benchmark_methods \
			feat \
			data/features/tiered_imagenet/test/feat_resnet12_tiered_imagenet.parquet.gzip \
			--config="resnet12_tiered_imagenet" \
			--n-shot=$${n_shot} \
			--device=${DEVICE} \
			--num-workers=${NUM_WORKERS}; \
	done

# Hyperparameter search
extract-mini-imagenet-val-features-with-resnet12:
	python -m scripts.predict_embeddings \
		feat_resnet12 \
		${MODEL_CHECKPOINTS_DIR}/feat_resnet12_mini_imagenet.pth \
		mini_imagenet \
		--split=val \
		--device=${DEVICE} \
		--num-workers=${NUM_WORKERS} \
		--batch-size=${BATCH_SIZE}

hyperparameter-search:
	for method in tim finetune pt_map laplacian_shot transductive_finetuning; do \
		python -m scripts.hyperparameter_search \
			$${method} \
			data/features/mini_imagenet/val/feat_resnet12_mini_imagenet.parquet.gzip \
			--n-shot=5 \
			--device=${DEVICE} \
			--num-workers=${NUM_WORKERS}; \
	done;
