import torch
import torchaudio
from dataclasses import dataclass
import math
from flask import Flask, render_template, request, jsonify
import base64
import io
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
import librosa
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()



app = Flask(__name__, static_folder='static')


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    log_likelihood: float = 0.0
    num_frames: int = 0

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0

        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        t -= 1
        if changed > stayed:
            j -= 1

        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        log_likelihood = sum(math.log(path[k].score) for k in range(i1, i2))
        num_frames = path[i2 - 1].time_index - path[i1].time_index + 1
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
                log_likelihood,
                num_frames
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * (seg.end - seg.start) for seg in segs) / sum(seg.end - seg.start for seg in segs)
                log_likelihood = sum(seg.log_likelihood for seg in segs)
                num_frames = sum(seg.num_frames for seg in segs) + 1
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score, log_likelihood, num_frames))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def process_audio(text, audio_file):
    global transcript, waveform
    transcript = format_sentence(text)
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio_file)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path)
    word_segments = merge_words(segments)

    # with open("output.txt", "w") as f:
    #     for i in range(len(word_segments)):
    #         ratio = waveform.size(1) / trellis.size(0)
    #         word = word_segments[i]
    #         x0 = int(ratio * word.start)

    #         # Determine the end time
    #         if i < len(word_segments) - 1:
    #             next_word = word_segments[i + 1]
    #             x1 = int(ratio * word.end)
    #         else:
    #             x1 = int(ratio * word.end)  # Last word segment, use its own end time

    #         start_time = x0 / bundle.sample_rate
    #         end_time = x1 / bundle.sample_rate
    #         f.write(f"{word.label}\t{start_time:.3f}\t{end_time:.3f}\t{word.log_likelihood:.2f}\t{word.score:.2f}\n")

def format_sentence(sentence):
    words = sentence.split()
    formatted_sentence = '|' + '|'.join(words) + '|'
    return formatted_sentence


@app.route('/')
def index():
    return render_template('index.html')

def merge_sentence(word_segments):
    sentence_score = sum(word.score * word.num_frames for word in word_segments) / sum(word.num_frames for word in word_segments)
    sentence_log_likelihood = sum(word.log_likelihood for word in word_segments)
    sentence = "".join([word.label for word in word_segments])
    return sentence, sentence_score, sentence_log_likelihood

@app.route('/process', methods=['POST'])
def process():
    audio_data = request.form['audio']
    text = request.form['text']

    audio_file = io.BytesIO(base64.b64decode(audio_data.split(',')[1]))

    global transcript, waveform
    transcript = format_sentence(text)
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    with torch.inference_mode():
        waveform, sample_rate = torchaudio.load(audio_file)

        if sample_rate != bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
            waveform = resampler(waveform)

        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path)
    word_segments = merge_words(segments)
    sentence, sentence_score, sentence_log_likelihood = merge_sentence(word_segments)

    results = {
        'word_segments': [],
        'segments': [],
        'sentence': sentence,
        'sentence_score': f'{sentence_score:.2f}',
        'sentence_log_likelihood': f'{sentence_log_likelihood:.2f}'
    }

    for i in range(len(word_segments)):
        ratio = waveform.size(1) / trellis.size(0)
        word = word_segments[i]
        x0 = int(ratio * word.start)

        if i < len(word_segments) - 1:
            next_word = word_segments[i + 1]
            x1 = int(ratio * word.end)
        else:
            x1 = int(ratio * word.end)

        start_time = x0 / bundle.sample_rate
        end_time = x1 / bundle.sample_rate
        results['word_segments'].append({
            'word': word.label,
            'start_time': f'{start_time:.3f}',
            'end_time': f'{end_time:.3f}',
            'log_likelihood': f'{word.log_likelihood:.2f}',
            'score': f'{word.score:.2f}'
        })

    for segment in segments:
        ratio = waveform.size(1) / trellis.size(0)
        x0 = int(ratio * segment.start)
        x1 = int(ratio * segment.end)
        start_time = x0 / bundle.sample_rate
        end_time = x1 / bundle.sample_rate
        results['segments'].append({
            'label': segment.label,
            'start_time': f'{start_time:.3f}',
            'end_time': f'{end_time:.3f}',
            'score': f'{segment.score:.2f}',
            'log_likelihood': f'{segment.log_likelihood:.2f}'
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)