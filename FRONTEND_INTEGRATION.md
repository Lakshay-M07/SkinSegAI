# SkinSegAI — Frontend Integration Contract

## 1. Backend status

The Python backend is implemented and verified.

- 96 automated regression tests passed.
- 14 additional subtests passed.
- 13 final live runtime checks passed.
- Actual ISIC validation sample tested.
- Local Ollama inference verified.
- Concurrent legacy and v2 HTTP requests verified.
- Session isolation and confirmation security verified.
- Report integrity verified.

The React frontend should migrate to the versioned v2 API.

Do not duplicate segmentation, AI routing, uncertainty analysis,
threshold analysis or evaluation logic in JavaScript.

## 2. Starting the backend

Activate the project virtual environment.

Install dependencies:

    python -m pip install -r requirements.txt

Ensure Ollama is running and the following model is installed:

    qwen3.5:9b-q4_K_M

Start FastAPI:

    python -m uvicorn backend.main:app \
      --host 127.0.0.1 \
      --port 8000 \
      --workers 1

API documentation:

    http://127.0.0.1:8000/docs

Versioned API base URL:

    http://127.0.0.1:8000/api/v2

Use the existing Vite API proxy during frontend development.

## 3. Endpoint catalogue

All paths below are relative to /api/v2.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /session | Create an assistant session |
| DELETE | /session | Delete a session |
| GET | /models | Available models and benchmarks |
| GET | /samples | Available curated samples |
| POST | /sample | Select a curated sample |
| POST | /image | Upload image and optional reference |
| GET | /image | Retrieve original image or reference |
| GET | /context | Current session state |
| POST | /configuration | Prepare manual configuration |
| POST | /chat | Natural-language assistant interaction |
| POST | /confirm | Approve or cancel segmentation |
| GET | /results | Numerical segmentation results |
| GET | /results/{model_id}/{view} | PNG visualization |
| GET | /analysis | Structured advanced analysis |
| GET | /history | Assistant conversation history |
| GET | /report | Download enhanced ZIP report |
| POST | /release | Release cached segmentation models |
| POST | /reset | Reset current session |

## 4. Session creation

POST /session?provider=ollama

Supported providers:

- ollama: local Qwen assistant.
- offline: deterministic commands and responses.

The default provider is offline.

For full natural-language functionality, explicitly request ollama.

Successful creation returns HTTP 201 with:

- session_id
- provider
- stage
- context

Store session_id in application state.

All session-specific requests require this header:

    X-Session-ID: <session_id>

The models and samples catalogue endpoints do not require it.

Treat the session identifier as a sensitive bearer token.
Do not include it in URLs or application logs.

The server supports two stored sessions and a 30-minute
inactivity timeout. Expired sessions must be recreated.

Sessions are not persisted across server restarts.

## 5. Revision tracking

The context object contains a revision integer.

Always use the latest revision returned by the backend.

Revision changes occur when:

- An image or sample is replaced.
- Active configuration changes.
- Segmentation successfully completes.
- Session results are cleared.
- The session is reset.

Configuration proposals do not change active settings or revision.

Do not increment revisions in JavaScript.

When a mutation succeeds, update the frontend from the returned
context and discard cached data belonging to older revisions.

For HTTP 409, refresh /context before allowing another operation.

## 6. Image upload

POST /image

Content-Type: multipart/form-data

Fields:

- image: required PNG or JPEG file.
- ground_truth: optional binary PNG reference mask.
- expected_revision: current revision integer.

Each uploaded file must not exceed 20 MiB.
Total upload request limit is 41 MiB.
Maximum decoded image size is 12 million pixels.

Reference masks must be single-channel, lossless PNG files
containing binary 0/1 or 0/255 values.

Reference dimensions must exactly match the image.

Do not manually set the multipart Content-Type header when
using browser FormData. Allow the browser to set its boundary.

Successful upload returns context, stage and image-quality
information.

Every new upload invalidates previous segmentation results
and pending approvals.

An old reference mask must never be automatically attached
to a newly selected image.

## 7. Curated samples

GET /samples

Returns:

    {
      "samples": [
        {
          "name": "ISIC_0000000",
          "has_ground_truth": true
        }
      ]
    }

POST /sample

JSON body:

    {
      "sample_id": "ISIC_0000000",
      "expected_revision": 0
    }

Replace the illustrative revision with the current one.

The backend loads the original sample and its matching
reference mask, if available.

If no samples are available, keep normal image upload enabled.

Do not assume the dataset is installed on every machine.

## 8. Original image and reference retrieval

GET /image?expected_revision=REVISION&view=original

GET /image?expected_revision=REVISION&view=reference

Both return binary image/png responses.

Use fetch with the X-Session-ID header, then convert the
response to a Blob and create an object URL.

Revoke obsolete object URLs when replacing images or
unmounting components.

A reference mask may legitimately be unavailable.

## 9. Available models

GET /models

Supported architecture identifiers:

- custom_unet
- resnet34_unet
- grabcut

Use these identifiers in API requests.

Use the returned model names and verified benchmark data
for display.

Do not hardcode benchmark statistics or replace unavailable
metrics with zero.

Benchmark results are aggregate dataset measurements and
must not be presented as individual image accuracy.

Timings from different hardware are not directly comparable.

## 10. Manual configuration

POST /configuration

Example complete JSON request:

    {
      "expected_revision": 1,
      "models": [
        "custom_unet",
        "resnet34_unet"
      ],
      "threshold": 0.5,
      "overlay_alpha": 0.35,
      "mc_passes": 1,
      "include_uncertainty": true,
      "include_morphology": true,
      "include_disagreement": true,
      "include_image_quality": true
    }

Supported ranges:

- models: one to three distinct supported models.
- threshold: 0.1 to 0.9.
- overlay_alpha: 0.0 to 1.0.
- mc_passes: integer from 1 to 20.

Multiple inference passes may increase runtime.
Stochastic estimation requires supported dropout layers.

GrabCut does not support neural prediction thresholds
or neural uncertainty maps.

Successful configuration returns:

- proposal
- stage
- provider
- context

The proposal contains the requested configuration, warnings,
confirmation_required and confirmation_token.

No segmentation has been executed at this point.

## 11. AI assistant conversation

POST /chat

JSON request:

    {
      "expected_revision": 1,
      "message": "Use both neural models with threshold 0.6."
    }

Response fields:

- message
- provider
- configured_provider
- used_fallback
- stage
- confirmation_required
- proposal
- execution_statistics
- context

Render the message as sanitized Markdown.

If confirmation_required is true, display the proposed settings
and explicit Approve and Cancel controls.

If proposal is null, do not assume an existing pending
confirmation has been cancelled.

Inform the user when used_fallback is true.

The local LLM selects validated tools. Python generates grounded
numerical explanations. This is not an unrestricted general
knowledge or medical diagnosis chatbot.

Maximum chat message length: 4000 characters.

## 12. Approve or cancel segmentation

POST /confirm

JSON request:

    {
      "confirmation_token": "<token from proposal>",
      "approved": true
    }

Use approved=false to cancel.

Requirements:

- Never automatically approve an AI proposal.
- Never execute segmentation on a slider movement alone.
- Do not treat natural-language approval as an API approval.
- Always use the most recently issued valid confirmation token.
- Disable duplicate approval clicks while the request is running.
- Never reuse consumed confirmation tokens.

A successful new proposal replaces the preceding token.

New image selection and session reset invalidate pending tokens.

Cancellation preserves previously committed settings and results.

After successful segmentation, obtain the new revision from
the returned context.

## 13. Segmentation results

GET /results

Response contains:

- revision
- image
- configuration
- results
- limitations

Each entry in results represents one selected algorithm.

Render the array dynamically, rather than assuming there
are always exactly two algorithms.

Result fields include:

- model_id
- algorithm
- predicted_lesion_pixels
- total_image_pixels
- predicted_lesion_area_percent
- mask_status
- inference_ms
- measurements
- ground_truth_metrics
- ground_truth_available
- probability_map_available
- uncertainty_map_available
- border_uncertainty
- available_views

Mask status can be empty, full or partial.

Ground-truth metrics contain Dice, IoU, precision and recall
when reference evaluation is available.

These metrics use a 0-to-1 scale.

Missing numerical values must be displayed as unavailable,
not zero.

Predicted lesion area is already expressed as a percentage.
Inference latency is expressed in milliseconds.

Only display morphology and uncertainty information when
enabled by configuration and available in the result.

## 14. Result visualizations

GET /results/{model_id}/{view}?expected_revision=REVISION

Possible views:

- mask
- overlay
- boundary
- uncertainty
- uncertainty_overlay
- error_map

Use each result's available_views array as the authoritative
list of supported visualizations.

Each endpoint returns binary PNG data.

Error-map colour legend:

- Green: true positive.
- Red: false positive.
- Blue: false negative.
- Black: true negative.

Do not request uncertainty views for GrabCut.

Do not request error maps without reference-mask availability.

Discard responses whose revision no longer matches the current
frontend revision.

## 15. Structured advanced analysis

GET /analysis?expected_revision=REVISION

Returns a revision and analysis object.

Relevant analysis sections:

- uncertainty
- model_agreement
- threshold_sensitivity
- image_quality

Respect availability flags and limitations in each section.

Threshold sensitivity is a preview using existing probability
maps. It does not change active configuration or rerun models.

Model agreement measures similarity, not correctness.

Predictive entropy is not calibrated confidence.

Pixel measurements do not establish physical lesion dimensions.

## 16. Conversation history

GET /history

Returns:

- revision
- history
- statistics

The backend conversation memory is bounded and revision-dependent.

Image replacement, configuration changes and completed
reanalysis can invalidate earlier backend conversation history.

The frontend may display earlier messages for readability,
but must mark obsolete analyses and proposals as historical.

Do not resubmit historical messages automatically.

## 17. Report download

GET /report?expected_revision=REVISION

Returns application/zip.

Recommended filename:

    SkinSegAI_Report.zip

Use authenticated fetch with the session header, then download
the response Blob.

The report includes segmentation images, measurements,
reference evaluation when available and enhanced analysis.

Report metadata includes report version and generation timestamp.

Report generation does not rerun inference or mutate session state.

## 18. Resource management

POST /release

JSON body:

    {
      "expected_revision": 1
    }

Releases cached segmentation models without removing existing
results or conversation state.

POST /reset

Uses the same revision request structure.

Clears the image, reference mask, results, conversation memory
and pending approvals.

DELETE /session

Deletes the current session and releases associated resources.

## 19. HTTP error handling

| Status | Frontend behaviour |
|---|---|
| 400 | Display invalid input error |
| 404 | Check whether session or requested resource is unavailable |
| 409 | Refresh context; invalidate stale actions |
| 413 | Display upload or request-size limit |
| 422 | Display request validation error |
| 429 | Session capacity reached |
| 503 | Backend, model or Ollama unavailable |

Do not automatically retry approval requests after network
errors. The operation may already have completed.

Fetch the current context before deciding whether to request
a new proposal.

An inference failure may leave the newly selected configuration
active while results are unavailable. Refresh context.

## 20. Frontend implementation requirements

Maintain a centralized application state for:

- Session identifier.
- Latest backend context and revision.
- Selected image source.
- Current configuration.
- Pending proposal and confirmation token.
- Segmentation results.
- Conversation messages.
- Active request and loading status.
- Visualization object URLs.
- Error messages.

Serialize state-changing requests.

Use AbortController and client-side request generation checks
to prevent delayed responses overwriting newer frontend state.

Aborting a browser request does not guarantee that backend
inference has stopped.

Do not automatically rerun segmentation while sliders move.
Prepare configuration and require explicit execution approval.

Keep the old results clearly marked until new results arrive,
or hide them when the active image or configuration changes.

Provide loading indicators for Ollama and segmentation.

Live chat responses may take several seconds.
Use an appropriate request timeout.

Keep model and benchmark displays dynamic.

Do not mix legacy segmentation results with v2 assistant state.

## 21. Recommended implementation order

1. Build the centralized v2 API client.
2. Implement session creation and recovery.
3. Connect image upload and curated sample selection.
4. Implement manual model configuration.
5. Implement explicit confirmation controls.
6. Render dynamic model results.
7. Connect PNG visualizations.
8. Implement the chatbot panel.
9. Connect advanced numerical analysis.
10. Implement ZIP report downloads.
11. Add loading, error and stale-state handling.
12. Perform browser-level end-to-end testing.

## 22. Verified runtime observations

On the tested Apple Silicon Mac with 16 GB unified memory:

- Local Qwen runtime footprint: 5.6 GB.
- Ollama GPU allocation: 100%.
- Context window: 4096 tokens.
- Final live average chat latency: 7.55 seconds.
- Maximum chat latency in that run: 13.30 seconds.
- Combined concurrent legacy/v2 request latency: 1.46 seconds.
- Lowest sampled system-wide memory availability: 18%.

These are observations from one runtime test, not guaranteed
performance limits.

## 23. Deployment limitations

This is a local academic application.

- Run one FastAPI worker.
- Keep the server bound to localhost.
- Session identifiers are not full user authentication.
- Session storage is in-memory.
- Operations are serialized to limit memory pressure.
- Memory limits are per request, not aggregate server guarantees.
- Large-image and prolonged multi-user workloads are not verified.
- Browser integration still requires acceptance testing.
- The application is not clinically validated.
- Segmentation is not medical diagnosis.

The legacy API remains available during the frontend migration.
The v2 API is the authoritative interface for the combined
manual and AI-assisted workflow.
