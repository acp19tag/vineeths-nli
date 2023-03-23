import tensorflow as tf


def get_segments(sentences):
    sentences_segments = []

    for sentence in sentences:
        temp = []
        i = 0

        for token in sentence.split(" "):
            temp.append(i)
            if token == "[SEP]":
                i += 1

        sentences_segments.append(temp)

    return sentences_segments


def get_inputs(input, maxlen, tokenizer):
    sentences = ["[CLS] " + " ".join(tokenizer.tokenize(sen1)) + " [SEP] "
                 + " ".join(tokenizer.tokenize(sen2)) + " [SEP] " for (sen1, sen2) in zip(input[0], input[1])]


    # If input is less than max length provided then the words are padded
    # ALSO if input is too long, then it is truncated TG ADDITION
    # sentences_padded = [sen + " [PAD]" * (maxlen - len(sen.split(" "))) if len(sen.split(" ")) != maxlen else sen
    #                     for sen in sentences]
    sentences_padded = [sen + " [PAD]" * (maxlen - len(sen.split(" "))) if len(sen.split(" ")) < maxlen else " ".join(sen.split(" ")[:maxlen])
                        for sen in sentences]
    
    # BERT requires a mask for the words which are padded.
    # For example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]
    sentences_mask = [[1] * len(sen.split(" ")) + [0] * (maxlen - len(sen.split(" "))) for sen in sentences_padded] # tg shifted from sentences to sentences_padded and moved below

    sentences_converted = [tokenizer.convert_tokens_to_ids(sen.split(" ")) for sen in sentences_padded]

    # For each separation [SEP], a new segment is converted
    sentences_segment = get_segments(sentences_padded)

    genLength = {len(sen.split(" ")) for sen in sentences_padded}
    
    print(f"len(sentences_converted) = {len(sentences_converted)}") # DEBUG
    print(f"len(sentences_segment) = {len(sentences_segment)}")
    print(f"len(sentences_mask) = {len(sentences_mask)}")
    print(f"len(sentences) = {len(sentences)}")
    print(f"len(sentences_padded) = {len(sentences_padded)}")
    print(f"len(genLength) = {len(genLength)}")

    if maxlen < 20:
        raise Exception("max length cannot be less than 20")
    elif len(genLength) != 1:
        print(genLength)
        raise Exception("sentences are not of same size")

    # Convert list into tensor integer arrays and return it
    return [tf.cast(sentences_converted, tf.int32), tf.cast(sentences_segment, tf.int32),
            tf.cast(sentences_mask, tf.int32)]