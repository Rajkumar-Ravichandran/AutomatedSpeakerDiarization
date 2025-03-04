# Speaker Diarization and Identification Pipeline

## Overview
This project implements an **automated speaker diarization and identification pipeline** that processes audio recordings to detect and label speakers. It leverages state-of-the-art **speech recognition, speaker embeddings, clustering, and speaker identification** techniques to analyze both clean and noisy audio samples.

## Features
- **Speech Transcription:** Uses OpenAI's Whisper model to convert speech into text.
- **Speaker Embedding Extraction:** Utilizes SpeechBrain's ECAPA-TDNN model to generate speaker embeddings.
- **Speaker Clustering:** Implements K-Means clustering to group speech segments based on speaker similarity.
- **Speaker Identification:** Matches clusters to known speakers using cosine similarity.
- **Performance Analysis:** Compares diarization results between clean and noisy audio, evaluating similarity scores and segment changes.

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install speechbrain==0.5.16 faster-whisper pyannote.audio whisper ctranslate2==4.4.0 librosa soundfile scikit-learn
```

## How It Works
1. **Transcribe Audio:**
   - Uses Whisper to segment speech into transcriptions with timestamps.
2. **Extract Speaker Embeddings:**
   - Uses SpeechBrain to extract speaker representations from audio segments.
3. **Cluster Speech Segments:**
   - Applies K-Means to group similar speaker embeddings.
4. **Assign Speaker Labels:**
   - Matches clustered embeddings with known speaker profiles based on cosine similarity.
5. **Analyze and Compare Results:**
   - Computes similarity scores, evaluates reliability of clusters, and assesses impact of noise.

## Usage
### 1. Load Required Models
```python
from speechbrain.pretrained import EncoderClassifier
embedding_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
```

### 2. Load Known Speaker Embeddings
```python
known_speaker_files = ["speaker_A.wav", "speaker_B.wav", "speaker_C.wav"]
known_speaker_embeddings = load_known_speaker_embeddings(known_speaker_files, embedding_model)
```

### 3. Run the Speaker Diarization Pipeline
```python
df_clean, n_segments_clean, centroids_clean, similarity_scores_clean = run_pipeline(
    "sample.wav", embedding_model, known_speaker_embeddings, n_clusters=3, whisper_model_name="base", similarity_threshold=0.7
)
```

### 4. Compare Clean vs Noisy Audio
```python
pct_increase = ((n_segments_noisy - n_segments_clean) / n_segments_clean) * 100
print(f"Percentage increase in segments from clean to noisy: {pct_increase:.2f}%")
```

## Results and Insights
- Computes the average **cosine similarity score** for speaker identification.
- Identifies clusters with high confidence and labels them.
- Assesses **how noise affects speaker recognition accuracy**.

## Acknowledgments
- [SpeechBrain](https://speechbrain.github.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyAnnote](https://github.com/pyannote/pyannote-audio)

## License
This project is open-source and free to use under the MIT License.


