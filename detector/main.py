import os
import pickle
from pathlib import Path
import transformers
from transformers import AutoTokenizer, TFAutoModel
from datasets import Dataset, DatasetDict, load_from_disk
from detector.utils import load_data
import tensorflow as tf
from keras import (layers, optimizers, callbacks,
                    Model, losses, metrics, Input, models)
import torch # needed for training huggingface models
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from joblib import dump, load


def prepare_datasets():
    train_size = int(os.environ.get("TRAIN_SIZE"))
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    gpt_path = cache_path / "gpt3_output"

    if not (cache_path / f"preprocessed_data_{train_size}").is_dir():
        print("🕑Creating datasets...\n")
        df_gpt3 = pd.read_csv(gpt_path / "gpt3_simple.csv").reset_index(drop=True)
        df_gpt3_advanced = pd.read_csv(gpt_path / "gpt3_advanced.csv").reset_index(drop=True)
        df_gpt3_advanced["version"] = "gpt3.5"
        df_gpt3_advanced.sample(frac=1,random_state=1).reset_index(drop=True)
        data = load_data()

        train, val, test = (data["train"].reset_index(drop=True),
                        data["valid"].reset_index(drop=True),
                        data["test"].reset_index(drop=True))


        def remove_newline(text: str) -> str:
            return text.replace("\n", " ")

        for df in [train, val, test]:
            df["version"] = np.where(df["AI"] ==1, "gpt2", "human")

        for df in [train, val, test, df_gpt3, df_gpt3_advanced]:
            df["text"] = df["text"].apply(remove_newline)
            df["text_length"] = df['text'].apply(len)

        train = pd.concat([train, df_gpt3.iloc[:140_000, :], df_gpt3_advanced.iloc[:-5000,:]])
        val = pd.concat([val, df_gpt3.iloc[140_000:145_000,:], df_gpt3_advanced.iloc[-5000:-2500,:]])
        test = pd.concat([test, df_gpt3.iloc[145_000:,:], df_gpt3_advanced.iloc[-2500:,:]])

        train_sample = pd.concat(
                        [train[train.version == "human"].sample(n=train_size//2, random_state=1),
                        train[train.version == "gpt2"].sample(n=train_size//8, random_state=1),
                        train[train.version == "gpt3"].sample(n=train_size//8, random_state=1),
                        train[train.version == "gpt3.5"].sample(n=train_size//4, random_state=1)]
                        ).reset_index(drop=True)
        val = pd.concat([val[val.version == "human"].sample(5000, random_state=1),
                        val[val.version == "gpt2"].sample(1250, random_state=1),
                        val[val.version == "gpt3"].sample(1250, random_state=1),
                        val[val.version == "gpt3.5"].sample(2500, random_state=1)]).reset_index(drop=True)
        test = pd.concat([test[test.version == "human"].sample(5000, random_state=1),
                        test[test.version == "gpt2"].sample(1250, random_state=1),
                        test[test.version == "gpt3"].sample(1250, random_state=1),
                        test[test.version == "gpt3.5"].sample(2500, random_state=1)]).reset_index(drop=True)

        ds_train = Dataset.from_pandas(train_sample, split="train")
        ds_val = Dataset.from_pandas(val, split="valid")
        ds_test = Dataset.from_pandas(test, split="test")
        ds_dict = DatasetDict({"train": ds_train, "valid": ds_val, "test": ds_test})
        ds_dict.save_to_disk(cache_path / f'preprocessed_data_{train_size}')
        print(f"✅Datasets created and saved in {cache_path}!\n")
    print("🕑Loading datasets from cache...\n")
    ds_dict = load_from_disk(cache_path / f'preprocessed_data_{train_size}')
    print("✅Datasets loaded from cache!\n")
    return ds_dict

def create_tokenizer(model_ckpt: str = 'google/electra-large-discriminator'):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    if not (cache_path / f"extractors_tokenizer_{model_ckpt.replace('/', '-')}").is_dir():
        print("🕑Creating tokenizer...\n")
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        tokenizer.save_pretrained(cache_path / f"extractors_tokenizer_{model_ckpt.replace('/', '-')}")
        print(f"✅Tokenizer saved in {cache_path}!\n")
    print("🕑Loading tokenizer from cache...\n")
    tokenizer = AutoTokenizer.from_pretrained(cache_path / f"extractors_tokenizer_{model_ckpt.replace('/', '-')}")
    print("✅Tokenizer loaded from cache!\n")
    return tokenizer

def encode_data(model_ckpt: str = 'google/electra-large-discriminator'):
    train_size = os.environ.get("TRAIN_SIZE")
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    if not (cache_path / f"encoded_data_{model_ckpt.replace('/', '-')}_{train_size}").is_dir():
        print("🕑Encoding data...\n")
        ds_dict = prepare_datasets()
        tokenizer = create_tokenizer(model_ckpt=model_ckpt)
        ds_encoded = ds_dict.map(tokenize, batched=True, batch_size=10_000)
        ds_encoded.save_to_disk(cache_path / f'encoded_data_{model_ckpt.replace("/", "-")}_{train_size}')
        print(f"✅Encoded data saved in {cache_path}!\n")
    print("🕑Loading encoded data from cache...\n")
    ds_encoded = load_from_disk(cache_path / f'encoded_data_{model_ckpt.replace("/", "-")}_{train_size}')
    print("✅Encoded data loaded from cache!\n")
    return ds_encoded

def instantiate_extractor(model_ckpt: str = 'google/electra-large-discriminator'):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    if not (cache_path / f"extractors_model_{model_ckpt.replace('/', '-')}").is_dir():
        print("🕑Instantiating extractor...\n")
        model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
        model.save_pretrained(cache_path / f"extractors_model_{model_ckpt.replace('/', '-')}")
        print(f"✅Extractor model saved in {cache_path}!\n")
    print("🕑Loading extractor model from cache...\n")
    model = TFAutoModel.from_pretrained(cache_path / f"extractors_model_{model_ckpt.replace('/', '-')}")
    print("✅Extractor model loaded from cache!\n")
    return model

def get_hidden_states(model_ckpt: str = 'google/electra-large-discriminator'):
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    train_size = os.environ.get("TRAIN_SIZE")
    def extract_hidden_states(batch):
            inputs = {k: v for k,v in batch.items() if k in tokenizer.model_input_names}
            last_hidden_state = model(**inputs).last_hidden_state
            return {"hidden_state": last_hidden_state[:, 0].numpy()}
    if not (cache_path / f"hidden_states_{model_ckpt.replace('/', '-')}_{train_size}").is_dir():
        print("🕑Extracting hidden states...\n")
        tokenizer = create_tokenizer(model_ckpt=model_ckpt)
        ds_encoded = encode_data(model_ckpt=model_ckpt)
        ds_encoded.set_format("tensorflow", columns=["input_ids", "attention_mask", "AI"])
        model = instantiate_extractor(model_ckpt=model_ckpt)
        ds_hidden = ds_encoded.map(extract_hidden_states, batched=True, batch_size=50)
        ds_hidden.save_to_disk(cache_path / f"hidden_states_{model_ckpt.replace('/', '-')}_{train_size}")
        print(f"✅Hidden states saved in {cache_path}!\n")
    print("🕑Loading hidden states from cache...\n")
    ds_hidden = load_from_disk(cache_path / f"hidden_states_{model_ckpt.replace('/', '-')}_{train_size}")
    print("✅Hidden states loaded from cache!\n")
    return ds_hidden

def train_model(model_ckpt: str = 'google/electra-large-discriminator',
                model_head: str = "lr",
                lr_C: list=[2**k for k in range(1,6)],
                ridge_alpha: list=[0.03, 0.035, 0.04, 0.045],
                nn_activation: str = "gelu",
                nn_layers: int = 2,
                nn_neurons: int = 64,
                nn_dropout: float = 0.1,
                nn_init_lr: float = 1e-3,
                nn_batch_size: int = 32):
    '''Train a model on the hidden states extracted from a pretrained model.
    ---
    model_ckpt: specifies the pretrained model to use for extracting hidden states
    ---
    model_head: specifies the model to use for classification. Options are
    'lr': "logistic regression"
    'ridge': "ridge classification
    'nn': "neural network.'''

    # perpare data
    cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
    train_size = os.environ.get("TRAIN_SIZE")
    ds_hidden = get_hidden_states(model_ckpt=model_ckpt)
    X_train = np.array(ds_hidden["train"]["hidden_state"])
    X_val = np.array(ds_hidden["valid"]["hidden_state"])
    X_test = np.array(ds_hidden["test"]["hidden_state"])

    y_train = np.array(ds_hidden["train"]["AI"])
    y_val = np.array(ds_hidden["valid"]["AI"])
    y_test = np.array(ds_hidden["test"]["AI"])
    X_search = np.vstack((X_train, X_val))
    y_search = np.hstack((y_train, y_val))
    split = PredefinedSplit([-1]*X_train.shape[0]+[0]*X_val.shape[0])

    Path(cache_path / f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}").mkdir(parents=True, exist_ok=True)
    Path(cache_path / f"trained_models_{train_size}").mkdir(parents=True, exist_ok=True)

    if not any((cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}").iterdir()):
        scores_dict = {"lr": 0,
                        "ridge": 0,
                        "nn": 0}
        with open(cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}/scores_dict.pkl", 'wb') as f:
            pickle.dump(scores_dict, f)
    else:
        with open(cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}/scores_dict.pkl", 'rb') as f:
            scores_dict = pickle.load(f)

    # train model
    if model_head == "lr" or model_head == "ridge":
        print(f"🕑Training model with model head {model_head}...\n")
        if model_head == "lr":
            lr_clf = LogisticRegression(max_iter=5000)
            params = {"C":lr_C}
            search = GridSearchCV(lr_clf,
                                param_grid=params,
                                n_jobs=-1,
                                cv = split,
                                scoring="accuracy")
            print("🕑Search in progress...\n")
            search.fit(X_search, y_search)
            print(f"👍🏽Best params: {search.best_params_}\n")
            best_model = search.best_estimator_
            print("🕑Fitting model...\n")
            best_model.fit(X_train, y_train)
            score = best_model.score(X_test, y_test)
            if score > scores_dict["lr"]:
                print(f"🟢New test score is {score} which is {score-scores_dict['lr']:.2f} better than previous best score!\n")
                scores_dict["lr"] = score
                with open(cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}/scores_dict.pkl", 'wb') as f:
                    pickle.dump(scores_dict, f)
                print("✅Best score saved!")
                dump(best_model, cache_path / f"trained_models_{train_size}" /
                    f"{model_head}_clf_best_{model_ckpt.replace('/', '-')}.joblib")
                print(f"✅Best model trained and saved in {cache_path}!\n")
            else:
                print(f"👎🏽New test score {score} is not better than previous best score!\n")

        if model_head == "ridge":
            ridge_clf = RidgeClassifierCV(alphas=ridge_alpha, cv=split)
            print("🕑Search in progress...\n")
            ridge_clf.fit(X_search, y_search)
            print(f"👍🏽Best params: {ridge_clf.alpha_}\n")
            best_model = ridge_clf
            score = best_model.score(X_test, y_test)
            if score > scores_dict["ridge"]:
                print(f"🟢New test score is {score} which is {score-scores_dict['ridge']:.2f} better than previous best score!\n")
                scores_dict["ridge"] = score
                with open(cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}/scores_dict.pkl", 'wb') as f:
                    pickle.dump(scores_dict, f)
                print("✅Best score saved!")
                dump(best_model, cache_path / f"trained_models_{train_size}" /
                    f"{model_head}_clf_best_{model_ckpt.replace('/', '-')}.joblib")
                print(f"✅Best model trained and saved in {cache_path}!\n")
            else:
                print(f"👎🏽New test score {score} is not better than previous best score!\n")

    elif model_head == "nn":
        print(f"🕑Training model with model head {model_head}...\n")
        nn_inputs = Input(shape=(X_train.shape[1],))
        for i in range(nn_layers):
            if i == 0:
                x = layers.Dense(nn_neurons, activation=nn_activation,
                                 kernel_initializer="he_normal")(nn_inputs)
                x = layers.Dropout(nn_dropout)(x)
            else:
                x = layers.Dense(nn_neurons, activation=nn_activation,
                                 kernel_initializer="he_normal")(x)
                x = layers.Dropout(nn_dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        nn_model = Model(inputs=nn_inputs, outputs = outputs)

        decay_steps = X_train.shape[0] // nn_batch_size

        lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=nn_init_lr,
                                                            decay_steps=decay_steps,
                                                            decay_rate=0.9)

        es = callbacks.EarlyStopping(monitor="val_binary_accuracy",
                                    mode="max",
                                    patience=10,
                                    restore_best_weights=True)

        nn_model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
                        metrics=[metrics.BinaryAccuracy()],
                        optimizer = optimizers.legacy.Adam(lr_schedule))
        print("🟢Model instantiated! Start training...\n")
        nn_model.fit(X_train, y_train, batch_size=nn_batch_size,
                        epochs=150,
                        validation_data=(X_val, y_val),
                        callbacks=[es])
        score = nn_model.evaluate(X_test, y_test)[1]
        if score > scores_dict["nn"]:
            print(f"🟢New test score is {score} which is {score-scores_dict['nn']:.4f} better than previous best score!\n")
            scores_dict["nn"] = score
            with open(cache_path/f"model_scores_{model_ckpt.replace('/', '-')}_{train_size}/scores_dict.pkl", 'wb') as f:
                pickle.dump(scores_dict, f)
            print("✅Best score saved!")
            best_model = nn_model
            best_model.save(cache_path/f"nn_model_{model_ckpt.replace('/', '-')}_{train_size}")
            print(f"✅Best model trained and saved in {cache_path}!\n")
        else:
            print(f"👎🏽New test score {score} is {scores_dict['nn']-score:.4f} worse than previous best score!\n")
    else:
        raise ValueError("❌model_head must be 'lr', 'ridge' or 'nn'!")
    return None

def get_demo_text(api: bool=False):
    '''returns a random text from the demo set'''
    if api:
        df_demo = pd.read_csv("models/gpt3_output/demo_data.csv").reset_index(drop=True)
    else:
        cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
        demo_path = cache_path / "gpt3_output" / "demo_data.csv"
        df_demo = pd.read_csv(demo_path).reset_index(drop=True)
    return df_demo.sample(n=1)["text"].values[0]

def get_prediction(text_input: str,
                   model_ckpt: str='google/electra-large-discriminator',
                   train_size: int=70_000,
                   model_head: str="nn",
                   api: bool=False) -> tuple[float, str]:
    '''outputs the probability of the text being AI written
    ---
    text_input: text to be classified
    ---
    model_ckpt: model to be used for feature extraction. Options are
    "google/electra-large-discriminator", and "roberta-large".
    ---
    train_size: size of the training set used for training the model. Options
    are 100_000, 70_000, and 40_000.
    ---
    model_head: model to be used for classification. Options are "nn"
    for neural network, "lr" for logistic regression, and
    "ridge" for ridge classifier.
    '''
    if not api:
        cache_path = Path(os.environ.get("LOCAL_REGISTRY_PATH"))
        # instantiate tokenizer and model
        tokenizer = create_tokenizer(model_ckpt=model_ckpt)
        model = instantiate_extractor(model_ckpt=model_ckpt)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(f"models/extractors_tokenizer_{model_ckpt.replace('/', '-')}"))
        model = TFAutoModel.from_pretrained(Path(f"models/extractors_model_{model_ckpt.replace('/', '-')}"))
    # extract features
    inputs = tokenizer(text_input.replace("\n", " "), return_tensors="tf")
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[:, 0].numpy()
    # choose model for classification and return prediction and probability
    proba = None
    class_pred = None
    if model_head == "nn":
        if api:
            nn_model = models.load_model(Path(f"models/nn_model_{model_ckpt.replace('/', '-')}_{train_size}"))
        else:
            if not (cache_path / f"nn_model_{model_ckpt.replace('/', '-')}_{train_size}").is_dir():
                print(f"🔴Model with {model_head} not found. Training model...\n")
                train_model(model_ckpt=model_ckpt, model_head=model_head)
            nn_model = models.load_model(cache_path/f"nn_model_{model_ckpt.replace('/', '-')}_{train_size}")
        proba = nn_model.predict(hidden_states, verbose=0)[0][0]
    elif model_head == "lr":
        if api:
            lr_clf_best = load(Path(f"models/trained_models_{train_size}/lr_clf_best_{model_ckpt.replace('/', '-')}.joblib"))
        else:
            if not (cache_path / f"trained_models_{train_size}/lr_clf_best_{model_ckpt.replace('/', '-')}.joblib").is_file():
                print(f"🔴Model with {model_head} not found. Training model...\n")
                train_model(model_ckpt=model_ckpt, model_head=model_head)
            lr_clf_best = load(f"{cache_path}/trained_models_{train_size}/lr_clf_best_{model_ckpt.replace('/', '-')}.joblib")
        proba = lr_clf_best.predict_proba(hidden_states)[0][1]
    elif model_head == "ridge":
        if api:
            ridge_clf = load(Path(f"models/trained_models_{train_size}/ridge_clf_best_{model_ckpt.replace('/', '-')}.joblib"))
        else:
            if not (cache_path / f"trained_models_{train_size}/ridge_clf_best_{model_ckpt.replace('/', '-')}.joblib").is_file():
                print(f"🔴Model with {model_head} not found. Training model...\n")
                train_model(model_ckpt=model_ckpt, model_head=model_head)
            ridge_clf = load(f"{cache_path}/trained_models_{train_size}/ridge_clf_best_{model_ckpt.replace('/', '-')}.joblib")
        decision = ridge_clf.decision_function(hidden_states)[0]
        proba = np.exp(decision)/(np.exp(-decision)+np.exp(decision))
    else:
        raise ValueError("❌Model must be one of 'nn', 'lr' or 'ridge'")
    if proba > 0.5:
        class_pred = "AI written"
    else:
        class_pred = "not AI written"
    print(f'Probability of the text input being AI written is {proba:.2f}. \n')
    print(f'The prediction therfore is that the text is {class_pred}.')
    return proba, class_pred
