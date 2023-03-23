# Parameters
MAX_SEQ_LEN = 128 # tg changed from 42 to 128 to 512 to 256
LSTM_UNITS = 64 # tg changed from 64 to 128

VOCAB_SIZE = 20_000 # tg changed from 20_000 to 200_000 to 40_000
EMBEDDING_DIM = 300
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
TRAIN_EMBED = False # tg changed from False to True

L2 = 4e-6
ACTIVATION = 'relu'
DROPOUT = 0.2
LEARNING_RATE = 0.001 # tg changed from 0.01
RHO = 0.9
EPSILON = 1e-08
DECAY = 0.0

CATEGORIES = 2 # TG changed from 3 to 2
BATCH_SIZE = 512 # tg changed from 512 to 256
TRAINING_EPOCHS = 100
VALIDATION_SPLIT = 0.02 # tg changed from 0.02 to 0.1

PATIENCE = 20 # tg changed from 4 to 20