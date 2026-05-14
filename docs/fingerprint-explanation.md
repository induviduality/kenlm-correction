### What is a "Fingerprint"?
In this architecture, a fingerprint is a highly structured, quantitative profile of a specific dysarthric patient's consistent speech errors. Because dysarthria causes predictable, mechanical alterations to speech (e.g., consistently substituting 'SH' with 'S' or dropping 'T' sounds), the fingerprint captures the mathematical frequency of these exact phonetic confusions.

### How It Works (New Architecture)
The pipeline uses a **sequence-level** comparison approach:
1. **Expected phonemes** are derived from the transcript text using g2p-en (no audio dependency)
2. **Observed phonemes** are decoded from the full audio using wav2vec2 CTC (no time-window slicing)
3. **Needleman-Wunsch alignment** matches the two sequences optimally
4. Each aligned position is classified as MATCH, SUB(stitution), DEL(etion), or INS(ertion)

This replaces the old forced-alignment approach (MMS_FA), which created false "silence" errors because rigid time windows didn't account for the irregular pacing of dysarthric speech.

### Understanding the Fingerprint JSON

| JSON Key | Description | How to Utilize for ASR/T5 Integration |
| :--- | :--- | :--- |
| **`speaker_id`** | The patient's ID (e.g., `"F01"`). | Use this to map the current ASR inference request to the correct pre-computed fingerprint. |
| **`severity`** | The clinical severity of the patient's dysarthria (e.g., `"mild"`, `"severe"`). | Can be used as a conditional control token for the model (e.g., `<\|severity_mild\|>`). |
| **`total_phoneme_observations`** | Total number of aligned phoneme positions evaluated. | Useful for setting confidence thresholds. Fingerprints with very low observations might be less reliable. |
| **`calibration_phrases`** | The transcript text of the utterances used for calibration. | Useful for auditing and debugging which sentences were analyzed. |
| **`error_map`** | A dictionary of specific errors and their occurrence frequency (e.g., `"SH>S": 0.67`). `SIL` means "Silence" (the phoneme was dropped/omitted). | **Direct Text Prompting:** Format the top $K$ errors into a text prefix string like `"Speaker Profile - Drops: T, R \| Substitutes: SH>S, DH>T"` and prepend it to the T5 error-correction model's input prompt. |
| **`pair_vocab`** | A fixed schema of 60 specific phoneme confusion pairs (e.g., `["T", "D"]`). | Acts as the legend/index mapping for the `fingerprint_vector`. |
| **`fingerprint_vector`** | A fixed-length array of 60 floats (e.g., `[0.18, 0.09, 0.10, ...]`) matching the `pair_vocab` order. | **Continuous Embedding Integration:** Because the vector is a fixed size (60 dimensions) representing the **Wilson Score Lower Bound** confidence of each error, you can pass this vector through a Linear layer and inject it directly into the Whisper model's decoder cross-attention blocks or as a learned prefix embedding for T5. |
| **`raw_error_counts`** / **`intended_phoneme_counts`** | Raw integer counts of how often an error happened vs. how often the intended phoneme was spoken. | Primarily used for analytical purposes or filtering out "noise" (e.g., an error that happened 1/1 times vs 50/50 times). |

### Example ASR Correction Workflow
1. User submits an audio file for Speaker `F01`.
2. The system loads `F01.json`.
3. **Whisper Phase:** Whisper transcribes the audio. Because F01 substitutes `SH` with `S` (`"SH>S": 0.67`), Whisper mistakenly transcribes "ship" as "sip".
4. **T5 Phase:** The T5 model receives the Whisper transcript ("sip") AND the `fingerprint_vector` (which signals a high probability of SH→S substitution).
5. **Correction:** The T5 model uses the fingerprint context to intelligently correct the output back to "ship".