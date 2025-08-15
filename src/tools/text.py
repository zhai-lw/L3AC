import jiwer

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.ReduceToListOfListOfWords()
])


def wer(prediction, ground_truth):
    return jiwer.wer(ground_truth, prediction, truth_transform=transformation, hypothesis_transform=transformation)


def wacc(prediction, ground_truth):
    score = 1 - wer(prediction, ground_truth)
    return score if score > 0. else 0.
