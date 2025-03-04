#This code implements an automated speaker diarization and identification pipeline that processes audio recordings to detect and label speakers. 
#It combines speech transcription, speaker embedding extraction, clustering, and speaker recognition to analyze clean and noisy audio samples.

#Speech Transcription: Uses the Whisper model to convert speech to text and segment the conversation.
#Speaker Embeddings: Extracts speaker features from each segment using SpeechBrain’s ECAPA-TDNN model.
#Clustering: Applies K-Means clustering to group segments based on speaker similarity.
#Speaker Identification: Matches clusters to known speakers by comparing embeddings using cosine similarity.
#Analysis: Compares clean vs. noisy audio in terms of:
#Speaker similarity scores
#Cluster reliability at different thresholds
#Segment count and total duration changes
#The final output provides insights into how noise affects speaker recognition accuracy and identifies high-confidence speaker clusters.


# Step 1: Install required libraries (already done in your environment)
# !pip install speechbrain==0.5.16 faster-whisper pyannote.audio whisper ctranslate2==4.4.0 librosa soundfile scikit-learn

# Step 2: Import necessary modules
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from pyannote.audio import Audio
from pyannote.core import Segment
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import traceback

# Step 3: Define helper functions

def convert_time(secs):
    """Convert seconds to hh:mm:ss format."""
    return str(timedelta(seconds=round(secs)))

def transcribe_audio(audio_file, model_name="base", language="en", beam_size=5, best_of=5):
    """Transcribe audio using Whisper model and return segments."""
    print(f"Transcribing {audio_file} with Whisper {model_name} model...")

    # Load Whisper model
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    # Transcribe audio
    segments, _ = model.transcribe(
        audio_file,
        language=language,
        beam_size=beam_size,
        best_of=best_of
    )

    # Format segments
    formatted_segments = []
    for segment in segments:
        formatted_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })

    print(f"Transcription completed. Found {len(formatted_segments)} segments.")
    return formatted_segments

def extract_segment_embedding(audio_file, segment, total_duration, embedding_model):
    """Extract speaker embedding for a specific segment."""
    try:
        # Initialize audio processor
        audio = Audio()

        # Extract start and end times from the segment
        start = segment['start']
        end = min(segment['end'], total_duration)  # Ensure end time doesn't exceed total duration

        # Create a segment object
        segment_obj = Segment(start, end)

        # Crop the waveform for the segment
        waveform, sample_rate = audio.crop(audio_file, segment_obj)

        # Convert to torch tensor if needed
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)

        # Make sure waveform has the right shape for the model
        # SpeechBrain expects [batch, time] format
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        # Get speaker embedding using the speechbrain model
        with torch.no_grad():
            embedding = embedding_model.encode_batch(waveform)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    except Exception as e:
        print(f"Error extracting embedding for segment {segment}: {e}")
        traceback.print_exc()
        return None

def compute_segment_embeddings(audio_file, segments, embedding_model):
    """Compute embeddings for all segments in the audio file."""
    # Load audio to get total duration
    waveform, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)

    # Initialize list to store embeddings
    embeddings = []

    # Process each segment
    for segment in segments:
        embedding = extract_segment_embedding(audio_file, segment, duration, embedding_model)
        if embedding is not None:
            embeddings.append(embedding)

    # Stack embeddings into a 2D array
    if embeddings:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.array([])

    return embeddings, duration

def cluster_embeddings(embeddings, n_clusters):
    """Cluster segment embeddings using KMeans."""
    # Initialize KMeans
    clustering = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit KMeans to embeddings
    clustering.fit(embeddings)

    # Get cluster labels and centroids
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    return labels, centroids

def compute_cluster_averages(embeddings, labels, n_clusters):
    """Calculate average embedding for each cluster."""
    # Initialize dictionary for cluster average embeddings
    cluster_avg_embeddings = {}

    # Calculate average embedding for each cluster
    for cluster_id in range(n_clusters):
        # Select embeddings belonging to this cluster
        cluster_embeddings = embeddings[labels == cluster_id]

        # Calculate mean embedding for the cluster
        avg_embedding = np.mean(cluster_embeddings, axis=0)

        # Store in dictionary
        cluster_avg_embeddings[cluster_id] = avg_embedding

    return cluster_avg_embeddings

def load_known_speaker_embeddings(known_speaker_files, embedding_model):
    """Load and compute embeddings for known speakers."""
    # Initialize dictionary for known speaker embeddings
    known_speaker_embeddings = {}

    # Process each known speaker file
    for file in known_speaker_files:
        try:
            # Extract speaker label from filename
            speaker_label = os.path.basename(file).split('.')[0]

            # Load audio file with librosa
            waveform, sample_rate = librosa.load(file, sr=16000, mono=True)

            # Convert to torch tensor
            waveform = torch.tensor(waveform).unsqueeze(0)

            # Compute embedding
            with torch.no_grad():
                embedding = embedding_model.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()

            # Store in dictionary
            known_speaker_embeddings[speaker_label] = embedding

        except Exception as e:
            print(f"Error processing known speaker file {file}: {e}")
            traceback.print_exc()

    print(f"Loaded embeddings for {len(known_speaker_embeddings)} known speakers.")
    return known_speaker_embeddings

def assign_speaker_labels(cluster_avg_embeddings, known_speaker_embeddings, similarity_threshold=0.7):
    """Assign speaker labels to clusters based on similarity to known speakers."""
    # Initialize dictionary for speaker assignments
    speaker_assignments = {}

    # Store the similarity scores for analysis
    similarity_scores = {}

    # For each cluster, find the best matching known speaker
    for cluster_id, avg_emb in cluster_avg_embeddings.items():
        best_score = -1
        best_speaker = None

        # Compare with each known speaker
        for speaker, known_emb in known_speaker_embeddings.items():
            # Reshape embeddings for cosine_similarity
            avg_emb_reshaped = avg_emb.reshape(1, -1)
            known_emb_reshaped = known_emb.reshape(1, -1)

            # Calculate cosine similarity
            similarity = cosine_similarity(avg_emb_reshaped, known_emb_reshaped)[0][0]

            # Update best match if this is better
            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker

        # Store the best similarity score
        similarity_scores[cluster_id] = best_score

        # Assign speaker label based on threshold
        if best_score >= similarity_threshold:
            speaker_assignments[cluster_id] = best_speaker
        else:
            speaker_assignments[cluster_id] = "Unknown"

    return speaker_assignments, similarity_scores

def run_pipeline(audio_file, embedding_model, known_speaker_embeddings, n_clusters=3, whisper_model_name="base", similarity_threshold=0.7):
    """Run the complete speaker diarization and identification pipeline."""
    print("\n" + "="*50)
    print(f"Processing audio file: {audio_file}")
    print("="*50)

    # Step 1: Transcribe audio
    segments = transcribe_audio(audio_file, model_name=whisper_model_name)

    # Step 2: Compute segment embeddings
    embeddings, duration = compute_segment_embeddings(audio_file, segments, embedding_model)

    # Step 3: Cluster embeddings
    if len(embeddings) > 0:
        labels, centroids = cluster_embeddings(embeddings, n_clusters)

        # Step 4: Compute average embeddings per cluster
        cluster_avg_embeddings = compute_cluster_averages(embeddings, labels, n_clusters)

        # Step 5: Assign speaker labels to clusters
        speaker_assignments, similarity_scores = assign_speaker_labels(
            cluster_avg_embeddings,
            known_speaker_embeddings,
            similarity_threshold
        )

        # Step 6: Annotate segments with cluster and speaker ID
        results = []
        for i, segment in enumerate(segments):
            if i < len(labels):  # Ensure we have a label for this segment
                cluster_id = int(labels[i])
                speaker_id = speaker_assignments[cluster_id]

                results.append({
                    "Start": convert_time(segment['start']),
                    "End": convert_time(segment['end']),
                    "Start_Seconds": segment['start'],
                    "End_Seconds": segment['end'],
                    "Text": segment['text'],
                    "Cluster": cluster_id,
                    "Speaker_ID": speaker_id
                })

        # Create DataFrame from results
        df = pd.DataFrame(results)

        # Print diarization results
        print("\nDiarization Results:")
        print(df[["Start", "End", "Cluster", "Speaker_ID"]])

        # Print similarity scores
        print("\nSimilarity Scores:")
        for cluster_id, score in similarity_scores.items():
            print(f"Cluster {cluster_id} -> Best similarity: {score:.3f}, Assigned to: {speaker_assignments[cluster_id]}")

        return df, len(segments), centroids, similarity_scores
    else:
        print("No embeddings were extracted. Pipeline cannot continue.")
        return pd.DataFrame(), len(segments), None, {}

# Step 4: Load models and process audio files

# Set device to CPU
device = "cpu"

# Load speaker embedding model
print("Loading speaker embedding model...")
embedding_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# Download and extract audio files if needed
# !wget https://drive.google.com/file/d/1ZcSTNF43seMPo0nF8sdXY5922Luc2bqn/view?usp=sharing -O wavs.tar.gz
# !tar -xvf wavs.tar.gz

# Load known speaker embeddings
known_speaker_files = [
    "speaker_A.wav",
    "speaker_B.wav",
    "speaker_C.wav",
    "speaker_D.wav",
    "speaker_E.wav"
]
known_speaker_embeddings = load_known_speaker_embeddings(known_speaker_files, embedding_model)

# Process clean audio sample
clean_audio = "sample.wav"
df_clean, n_segments_clean, centroids_clean, similarity_scores_clean = run_pipeline(
    clean_audio,
    embedding_model,
    known_speaker_embeddings,
    n_clusters=3,
    whisper_model_name="base",
    similarity_threshold=0.7
)

# Process noisy audio sample
noisy_audio = "sample_noisy.wav"
df_noisy, n_segments_noisy, centroids_noisy, similarity_scores_noisy = run_pipeline(
    noisy_audio,
    embedding_model,
    known_speaker_embeddings,
    n_clusters=3,
    whisper_model_name="base",
    similarity_threshold=0.7
)

# Step 5: Infer targeted insights

# Calculate average cosine similarity score for clean audio
avg_sim_clean = sum(similarity_scores_clean.values()) / len(similarity_scores_clean)
print(f"\n Average cosine similarity score for clean audio: {avg_sim_clean:.3f}")

# Calculate average cosine similarity score for noisy audio
avg_sim_noisy = sum(similarity_scores_noisy.values()) / len(similarity_scores_noisy)
print(f"\n Average cosine similarity score for noisy audio: {avg_sim_noisy:.3f}")

# Count clusters in clean sample that meet/exceed threshold of 0.7
reliable_clean = sum(1 for score in similarity_scores_clean.values() if score >= 0.7)
print(f"\n Clusters in clean sample with similarity >= 0.7: {reliable_clean}")

# Count clusters in noisy sample that meet/exceed threshold of 0.7
reliable_noisy = sum(1 for score in similarity_scores_noisy.values() if score >= 0.7)
print(f"\n Clusters in noisy sample with similarity >= 0.7: {reliable_noisy}")

# Calculate percentage increase in segments from clean to noisy
pct_increase = ((n_segments_noisy - n_segments_clean) / n_segments_clean) * 100
print(f"\n Percentage increase in segments from clean to noisy: {pct_increase:.2f}%")

# Calculate total duration for clean sample
total_duration_clean = sum(df_clean["End_Seconds"] - df_clean["Start_Seconds"])
print(f"\n Total duration (seconds) for clean sample: {total_duration_clean:.2f}")

# Check which cluster would be assigned at threshold 0.65
print("\n Checking which cluster would be assigned at threshold 0.65 but not at 0.7:")
for cluster_id, score in similarity_scores_noisy.items():
    if 0.65 <= score < 0.7:
        print(f"Cluster {cluster_id} has score {score:.3f}")

# Find highest confidence cluster in clean sample
max_cluster_clean = max(similarity_scores_clean.items(), key=lambda x: x[1])
cluster_id, max_score = max_cluster_clean
speaker_label = next((k for k, v in df_clean.groupby("Cluster")["Speaker_ID"].first().items() if k == cluster_id), None)
print(f"\n Highest confidence cluster in clean sample: Cluster {cluster_id} → {speaker_label} with score {max_score:.3f}")
