"use client";

/**
 * Uploads a media file, polls the job, and shows the cleaned SRT.
 *   <SubtitleDemo apiUrl="https://api.your-site.com" />
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import styles from "./SubtitleDemo.module.css";

export interface SubtitleDemoProps {
  /** Base URL of the API, with no trailing slash. */
  apiUrl: string;
  /** Sent as X-API-Key when the server is configured to require one. */
  apiKey?: string;
  /** "tr" or "en". Defaults to Turkish. */
  defaultLanguage?: "tr" | "en";
  /** Show the model selector (only useful if ALLOW_MODEL_OVERRIDE=true). */
  allowModelChoice?: boolean;
  className?: string;
}

type Phase = "idle" | "uploading" | "working" | "done" | "error";

interface JobStatus {
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  duration: number | null;
  queue_position: number;
  cue_count: number;
  error: string | null;
  srt_ready: boolean;
}

const ACCEPT = ".mp4,.mov,.mkv,.avi,.webm,.mp3,.wav,.m4a,.ogg,.flac";
const POLL_MS = 1000;

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function SubtitleDemo({
  apiUrl,
  apiKey,
  defaultLanguage = "tr",
  allowModelChoice = false,
  className,
}: SubtitleDemoProps) {
  const base = apiUrl.replace(/\/$/, "");

  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<"tr" | "en">(defaultLanguage);
  const [model, setModel] = useState("large-v3");
  const [phase, setPhase] = useState<Phase>("idle");
  const [uploadPct, setUploadPct] = useState(0);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [srt, setSrt] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      // Stop polling and cancel any in-flight upload when unmounted.
      aliveRef.current = false;
      if (pollRef.current) clearTimeout(pollRef.current);
      xhrRef.current?.abort();
    };
  }, []);

  const headers = useCallback((): HeadersInit => {
    return apiKey ? { "X-API-Key": apiKey } : {};
  }, [apiKey]);

  const reset = useCallback(() => {
    if (pollRef.current) clearTimeout(pollRef.current);
    xhrRef.current?.abort();
    setPhase("idle");
    setUploadPct(0);
    setJob(null);
    setJobId(null);
    setSrt("");
    setError("");
    setCopied(false);
  }, []);

  const selectFile = useCallback(
    (next: File | null) => {
      reset();
      setFile(next);
    },
    [reset],
  );

  const poll = useCallback(
    async (id: string) => {
      if (!aliveRef.current) return;
      try {
        const res = await fetch(`${base}/api/jobs/${id}`, { headers: headers() });
        if (!res.ok) throw new Error(`Job lookup failed (${res.status})`);
        const status: JobStatus = await res.json();
        if (!aliveRef.current) return;
        setJob(status);

        if (status.status === "error") {
          setError(status.error || "Transcription failed.");
          setPhase("error");
          return;
        }
        if (status.status === "done") {
          const srtRes = await fetch(`${base}/api/jobs/${id}/srt`, {
            headers: headers(),
          });
          if (!srtRes.ok) throw new Error(`Could not fetch the SRT (${srtRes.status})`);
          const text = await srtRes.text();
          if (!aliveRef.current) return;
          setSrt(text);
          setPhase("done");
          return;
        }
        pollRef.current = setTimeout(() => poll(id), POLL_MS);
      } catch (err) {
        if (!aliveRef.current) return;
        setError(err instanceof Error ? err.message : String(err));
        setPhase("error");
      }
    },
    [base, headers],
  );

  const start = useCallback(() => {
    if (!file) return;
    setPhase("uploading");
    setUploadPct(0);
    setError("");
    setSrt("");
    setJob(null);

    const form = new FormData();
    form.append("file", file);
    form.append("language", language);
    if (allowModelChoice) form.append("model", model);

    // XHR rather than fetch: only XHR reports upload progress.
    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr;
    xhr.open("POST", `${base}/api/transcribe`);
    if (apiKey) xhr.setRequestHeader("X-API-Key", apiKey);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setUploadPct(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (!aliveRef.current) return;
      let payload: any = {};
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error body */
      }
      if (xhr.status >= 200 && xhr.status < 300 && payload.job_id) {
        setJobId(payload.job_id);
        setPhase("working");
        poll(payload.job_id);
      } else {
        setError(payload.detail || `Upload failed (${xhr.status})`);
        setPhase("error");
      }
    };

    xhr.onerror = () => {
      if (!aliveRef.current) return;
      setError("Network error. Is the API reachable and CORS configured?");
      setPhase("error");
    };

    xhr.send(form);
  }, [file, language, model, allowModelChoice, base, apiKey, poll]);

  const download = useCallback(() => {
    const blob = new Blob([srt], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stem = (file?.name || "subtitles").replace(/\.[^.]+$/, "");
    link.href = url;
    link.download = `${stem}_${language}.srt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [srt, file, language]);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(srt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Could not copy to the clipboard.");
    }
  }, [srt]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) selectFile(dropped);
    },
    [selectFile],
  );

  const busy = phase === "uploading" || phase === "working";
  const pct = phase === "uploading" ? uploadPct : Math.round((job?.progress ?? 0) * 100);

  let statusLine = "";
  if (phase === "uploading") {
    statusLine = `Uploading... ${uploadPct}%`;
  } else if (phase === "working" && job) {
    if (job.status === "queued") {
      statusLine =
        job.queue_position > 1
          ? `Queued - position ${job.queue_position}`
          : "Queued - starting shortly";
    } else if (job.stage === "Transcribing" && job.duration) {
      statusLine = `Transcribing... ${pct}% (${formatClock(
        job.progress * job.duration,
      )} / ${formatClock(job.duration)})`;
    } else {
      statusLine = job.stage;
    }
  } else if (phase === "done" && job) {
    statusLine = `Done - ${job.cue_count} subtitles`;
  }

  return (
    <div className={[styles.root, className].filter(Boolean).join(" ")}>
      <div className={styles.header}>
        <h2 className={styles.title}>Whisper Subtitle Generator</h2>
        <span className={styles.badge}>large-v3 · GPU</span>
      </div>

      <label
        className={[styles.dropzone, dragging ? styles.dropzoneActive : ""]
          .filter(Boolean)
          .join(" ")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept={ACCEPT}
          className={styles.fileInput}
          disabled={busy}
          onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <>
            <strong className={styles.fileName}>{file.name}</strong>
            <span className={styles.hint}>{formatBytes(file.size)} · click to change</span>
          </>
        ) : (
          <>
            <strong className={styles.fileName}>Drop a video or audio file</strong>
            <span className={styles.hint}>or click to browse · mp4, mkv, mp3, wav…</span>
          </>
        )}
      </label>

      <div className={styles.controls}>
        <div className={styles.control}>
          <span className={styles.label}>Language</span>
          <div className={styles.segmented} role="group">
            {(["tr", "en"] as const).map((code) => (
              <button
                key={code}
                type="button"
                disabled={busy}
                onClick={() => setLanguage(code)}
                className={[
                  styles.segment,
                  language === code ? styles.segmentActive : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                {code === "tr" ? "Turkish" : "English"}
              </button>
            ))}
          </div>
        </div>

        {allowModelChoice && (
          <div className={styles.control}>
            <span className={styles.label}>Model</span>
            <select
              className={styles.select}
              value={model}
              disabled={busy}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="large-v3">large-v3 (most accurate)</option>
              <option value="large-v3-turbo">large-v3-turbo (faster)</option>
              <option value="medium">medium</option>
            </select>
          </div>
        )}
      </div>

      <div className={styles.actions}>
        <button
          type="button"
          className={styles.primary}
          onClick={start}
          disabled={!file || busy}
        >
          {busy ? "Working..." : "Generate Subtitles"}
        </button>
        {(phase === "done" || phase === "error") && (
          <button type="button" className={styles.secondary} onClick={reset}>
            Reset
          </button>
        )}
      </div>

      {(busy || phase === "done") && (
        <div className={styles.progressWrap}>
          <div className={styles.progressTrack}>
            <div
              className={[
                styles.progressBar,
                phase === "working" && job?.status === "queued"
                  ? styles.progressIndeterminate
                  : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={
                phase === "working" && job?.status === "queued"
                  ? undefined
                  : { width: `${pct}%` }
              }
            />
          </div>
          <span className={styles.status}>{statusLine}</span>
        </div>
      )}

      {error && <div className={styles.error}>{error}</div>}

      {phase === "done" && srt && (
        <div className={styles.result}>
          <div className={styles.resultHead}>
            <span className={styles.label}>Cleaned SRT</span>
            <div className={styles.resultActions}>
              <button type="button" className={styles.secondary} onClick={copy}>
                {copied ? "Copied" : "Copy"}
              </button>
              <button type="button" className={styles.primary} onClick={download}>
                Download .srt
              </button>
            </div>
          </div>
          <pre className={styles.srt}>{srt}</pre>
        </div>
      )}
    </div>
  );
}
