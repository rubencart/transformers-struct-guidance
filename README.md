# Structural Guidance for Transformer Language Models + Transformer Grammars

This is a fork of the [transformers-struct-guidance](https://github.com/IBM/transformers-struct-guidance) repository,
to which we added code that we used to train larger models, as well as our version of the Transformer-Grammars model.
Official code for this last model was later released too, see [transformer_grammars](https://github.com/google-deepmind/transformer_grammars), but
it wasn't available at the time of our study.
The rest of the code for our study is available in the [USCOCO](https://github.com/rubencart/USCOCO) repository.

The `scripts/` folder contains various script that show how to run the code to train models, after
having done the installation and data preparation as described below.

Original readme:

# Structural Guidance for Transformer Language Models

This repository accompanies the paper, [Structural Guidance for Transformer Language Models](https://arxiv.org/pdf/2108.00104.pdf), published in ACL 2021. It includes inplementation of parsing-as-language-modelling and structural scaffolding for Transformer language models.

## Environment

```
conda create -n syntra python=3.6
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch -c conda-forge
bash scripts/download_and_patch_transformers.sh
pip install -e src/transformers/
pip install nltk
python -c "import nltk;nltk.download('punkt')"
```

The code is based on Python3. You can install the different modules with
```
bash scripts/download_and_patch_transformers.sh
pip install -r requirements.txt
python -c "import nltk;nltk.download('punkt')"
```

The Huggingface transformers is updated indirectly through a patch. If you
modifiy the code, to commit changes run

```
bash scripts/generate_patch.sh
```

and then just commit this patch


## Data preparation

### Prepare parsing oracle files 

PLM and ScLM require syntactic parses to derive the action sequence oracle. The following command demonstrates how to prepare oracle files for these models.

```
python src/get_oracle.py --gen --fpath train.txt > train_gen.oracle
python src/get_oracle.py --gen --fpath dev.txt > dev_gen.oracle
python src/get_oracle.py --gen --fpath test.txt > test_gen.oracle
```

### Prepare action ngram list

The following command generates the action ngram list for ScLM models. The training code of ScLM assumes that the action ngram list is stored in the root folder.

```
python src/get_action_ngram_list.py -f path/to/bllip-lg_train_gen.oracle path/to/bllip-lg_dev_gen.oracle -o bllip-lg_action_ngram_list.txt
```

## Vanilla Language Models (LM)

The script `src/lm.py` implements a vanilla Transformer language model. Below are the commands for model training and evaluation, as well as commands to compute word-level surprisals from a trained model.

```
# Model training
python src/lm.py --train_data train.txt --dev_data dev.txt --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --random_init --batch_size ${BATCH_SIZE} --report ${REPORT} --sample_every ${SAMPLE_EVERY} --model_path ${MODEL_PATH}

# Compute word-level perplexity
python src/lm.py --restore_from ${MODEL_PATH} --test_data test.txt --do_test

# Estimate word surprisals
python src/lm.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

## Scaffoled Language Models (ScLM)

The script `src/lm-sc.py` implements Transformer language model with structural prediction as an auxilliary task, referred as ScLM. The commanline variable, ${SCAFFOLD_TYPE}, can be set as `past` or `next`, which corresponds to `ScLM-past` or `ScLM-next` respectively in the paper.

```
# Model training  
python src/lm-sc.py --train_data train_gen.oracle --dev_data dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --random_init --batch_size ${BATCH_SIZE} --report ${REPORT} --sample_every ${SAMPLE_EVERY} --alpha 0.5 --scaffold_type ${SCAFFOLD_TYPE} --model_path ${MODEL_PATH}

# Compute word-level perplexity
python src/plm-gen.py --restore_from ${MODEL_PATH} --test_data test_gen.oracle --do_test

# Estimate word surprisals
python src/lm-sc.py --restore_from ${MODEL_PATH} --do_eval --fpath ${TEST_SUITE_PATH} --pretokenized > ${OUTPUT_PATH}
```

## Parsing as Language Modelling (PLM/PLM-mask)

The script `src/plm-gen.py` implements the idea of generative parsing as language modelling, a probabilistic model of top-down parsing action sequence. There are two variants, PLM and PLM-mask.

For PLM:
```
# Model training for PLM
python src/plm-gen.py --train_data train_gen.oracle --dev_data dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --batch_size ${BATCH_SIZE} --random_init --report ${REPORT} --sample_every ${SAMPLE_EVERY} --model_path ${MODEL_PATH}

# Estimate word-level perplexity with PLM
python src/plm-gen.py --restore_from ${MODEL_PATH} --test_data test_gen.oracle --do_test

# Estimate word surprisals with PLM
python src/plm-gen.py --restore_from ${MODEL_PATH} --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_SUITE_PATH} > ${OUTPUT_PATH} 2>${EVAL_LOG_PATH}
```

For PLM-mask:
```
# Model training for PLM-mask
python src/plm-gen.py --train_data train_gen.oracle --dev_data dev_gen.oracle --lr 1e-5 --epochs ${EPOCHS} --seed ${SEED} --do_train --batch_size ${BATCH_SIZE} --random_init --add_structured_mask --buffer_head 0 --stack_head 1 --report ${REPORT} --sample_every ${SAMPLE_EVERY} --model_path ${MODEL_PATH}

# Estimate word-level perplexity with PLM-mask
python src/plm-gen.py --restore_from ${MODEL_PATH} --add_structured_mask --buffer_head 0 --stack_head 1 --test_data test_gen.oracle --do_test

# Estimate word surprisals with PLM-mask
python src/plm-gen.py --restore_from ${MODEL_PATH} --add_structured_mask --buffer_head 0 --stack_head 1 --do_eval --beam_size 100 --word_beam_size 10 --fast_track_size 5 --pretokenized --fpath ${TEST_SUITE_PATH} > ${OUTPUT_PATH} 2>>${EVAL_LOG_PATH}
```

## Plot figures

The `analysis` folder contains the code and model evaluation results for generating the figures in the paper. The following commands run the plotting scripts and generate figures in the `figs` folder. Python packages `matplotlib` and `pandas` are required to run the plotting scripts. RNNG results are taken from [Hu et al., (2020)](https://github.com/cpllab/syntactic-generalization/tree/master/data/raw/sg_results).

```
cd analysis
mkdir -p figs

# Plot results on SG Test Suites and BLiMP-10%.
python analysis_sg.py
python analysis_blimp.py
```

## Acknowledgements

We thank Ramon Astudillo and Tahira Naseem for their contributions to the repository.
