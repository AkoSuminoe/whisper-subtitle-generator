"use client";

/**
 * Uploads a media file, polls the job, and shows the cleaned SRT.
 *   <SubtitleDemo apiUrl="https://api.your-site.com" turnstileSiteKey="0x4AAA..." />
 */

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./SubtitleDemo.module.css";

export interface SubtitleDemoProps {
  /** Base URL of the API, with no trailing slash. */
  apiUrl: string;
  /** Cloudflare Turnstile site key. Omit only if the server has no secret set. */
  turnstileSiteKey?: string;
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
  eta_seconds: number;
  cue_count: number;
  error: string | null;
}

interface Limits {
  max_upload_mb: number;
  max_duration_sec: number;
  rate_limit_per_hour: number;
}

const ACCEPT = ".mp4,.mov,.mkv,.avi,.webm,.mp3,.wav,.m4a,.ogg,.flac";
const POLL_MS = 1000;
const TURNSTILE_SRC = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

declare global {
  interface Window {
    turnstile?: {
      render: (el: HTMLElement, opts: Record<string, unknown>) => string;
      reset: (id?: string) => void;
      remove: (id?: string) => void;
    };
  }
}

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

/** Load the Turnstile script once per page. */
function loadTurnstile(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.turnstile) return Promise.resolve();
  const existing = document.querySelector(`script[src="${TURNSTILE_SRC}"]`);
  if (existing) {
    // The tag may already have loaded, in which case "load" never fires again.
    return new Promise((resolve) => {
      if (window.turnstile) return resolve();
      existing.addEventListener("load", () => resolve());
      const poll = setInterval(() => {
        if (window.turnstile) {
          clearInterval(poll);
          resolve();
        }
      }, 100);
      setTimeout(() => clearInterval(poll), 15000);
    });
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = TURNSTILE_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Could not load the captcha."));
    document.head.appendChild(script);
  });
}

export default function SubtitleDemo({
  apiUrl,
  turnstileSiteKey,
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
  const [srt, setSrt] = useState("");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);
  const [token, setToken] = useState("");
  const [limits, setLimits] = useState<Limits | null>(null);
  const [termsText, setTermsText] = useState("");
  const [termsError, setTermsError] = useState("");
  const [termsOpen, setTermsOpen] = useState(false);

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const aliveRef = useRef(true);
  const captchaRef = useRef<HTMLDivElement | null>(null);
  const widgetRef = useRef<string | null>(null);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      if (pollRef.current) clearTimeout(pollRef.current);
      xhrRef.current?.abort();
    };
  }, []);

  // Advertise the server's real limits rather than hard-coding them here.
  useEffect(() => {
    fetch(`${base}/api/health`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (aliveRef.current && data?.limits) setLimits(data.limits);
      })
      .catch(() => undefined);
  }, [base]);

  useEffect(() => {
    if (!turnstileSiteKey || !captchaRef.current) return;
    let cancelled = false;
    loadTurnstile()
      .then(() => {
        if (cancelled || !captchaRef.current || !window.turnstile) return;
        widgetRef.current = window.turnstile.render(captchaRef.current, {
          sitekey: turnstileSiteKey,
          callback: (t: string) => setToken(t),
          "expired-callback": () => setToken(""),
          "error-callback": () => setToken(""),
          theme: "auto",
        });
      })
      .catch(() => setError("Could not load the captcha. Check your connection."));
    return () => {
      cancelled = true;
      if (widgetRef.current && window.turnstile) {
        window.turnstile.remove(widgetRef.current);
        widgetRef.current = null;
      }
    };
  }, [turnstileSiteKey]);

  const headers = useCallback(
    (): HeadersInit => (apiKey ? { "X-API-Key": apiKey } : {}),
    [apiKey],
  );

  const resetCaptcha = useCallback(() => {
    setToken("");
    if (widgetRef.current && window.turnstile) window.turnstile.reset(widgetRef.current);
  }, []);

  // Validate as the user types: a bad dictionary should be obvious before upload.
  const onTermsChange = useCallback((value: string) => {
    setTermsText(value);
    const trimmed = value.trim();
    if (!trimmed) {
      setTermsError("");
      return;
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        setTermsError('Must be an object, like {"wrong": "correct"}.');
        return;
      }
      const bad = Object.entries(parsed).find(([, v]) => typeof v !== "string");
      setTermsError(bad ? `"${bad[0]}" must map to text.` : "");
    } catch {
      setTermsError("Not valid JSON yet.");
    }
  }, []);

  const reset = useCallback(() => {
    if (pollRef.current) clearTimeout(pollRef.current);
    xhrRef.current?.abort();
    resetCaptcha();
    setPhase("idle");
    setUploadPct(0);
    setJob(null);
    setSrt("");
    setError("");
    setCopied(false);
    setTermsText("");
    setTermsError("");
  }, [resetCaptcha]);

  const selectFile = useCallback(
    (next: File | null) => {
      if (next && limits && next.size > limits.max_upload_mb * 1024 * 1024) {
        setError(`File is too large. The limit is ${limits.max_upload_mb} MB.`);
        return;
      }
      reset();
      setFile(next);
    },
    [reset, limits],
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
          const srtRes = await fetch(`${base}/api/jobs/${id}/srt`, { headers: headers() });
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
    if (turnstileSiteKey && !token) {
      setError("Please complete the captcha first.");
      return;
    }
    setPhase("uploading");
    setUploadPct(0);
    setError("");
    setSrt("");
    setJob(null);

    const form = new FormData();
    form.append("file", file);
    form.append("language", language);
    form.append("turnstile_token", token);
    if (termsText.trim()) form.append("terms", termsText.trim());
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
      // The token is single-use, so a new one is needed either way.
      resetCaptcha();
      if (xhr.status >= 200 && xhr.status < 300 && payload.job_id) {
        setPhase("working");
        poll(payload.job_id);
      } else {
        setError(payload.detail || `Upload failed (${xhr.status})`);
        setPhase("error");
      }
    };

    xhr.onerror = () => {
      if (!aliveRef.current) return;
      resetCaptcha();
      setError("Network error. Is the API reachable and CORS configured?");
      setPhase("error");
    };

    xhr.send(form);
  }, [
    file, language, model, allowModelChoice, base, apiKey, poll, token,
    turnstileSiteKey, resetCaptcha, termsText,
  ]);

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

  const busy = phase === "uploading" || phase === "working";
  const queued = phase === "working" && job?.status === "queued";
  const pct = phase === "uploading" ? uploadPct : Math.round((job?.progress ?? 0) * 100);

  let statusLine = "";
  if (phase === "uploading") {
    statusLine = `Uploading... ${uploadPct}%`;
  } else if (phase === "working" && job) {
    if (job.status === "queued") {
      const wait = job.eta_seconds > 0 ? ` · about ${formatClock(job.eta_seconds)} left` : "";
      statusLine = `In line — position ${job.queue_position}${wait}`;
    } else if (job.stage === "Transcribing" && job.duration) {
      statusLine = `Transcribing... ${pct}% (${formatClock(
        job.progress * job.duration,
      )} / ${formatClock(job.duration)})`;
    } else {
      statusLine = job.stage;
    }
  } else if (phase === "done" && job) {
    statusLine = `Done — ${job.cue_count} subtitles`;
  }

  const canStart =
    !!file && !busy && !termsError && (!turnstileSiteKey || !!token);

  return (
    <div className={[styles.root, className].filter(Boolean).join(" ")}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Subtitle Generator</h2>
          <p className={styles.subtitle}>Video or audio in, clean SRT out.</p>
        </div>
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
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) selectFile(dropped);
        }}
      >
        <input
          type="file"
          accept={ACCEPT}
          className={styles.fileInput}
          disabled={busy}
          onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
        />
        <span className={styles.dropIcon} aria-hidden="true">
          {file ? "🎬" : "⬆"}
        </span>
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

      {limits && (
        <p className={styles.limits}>
          Up to {limits.max_upload_mb} MB and {Math.round(limits.max_duration_sec / 60)}{" "}
          minutes · {limits.rate_limit_per_hour} per hour
        </p>
      )}

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
                className={[styles.segment, language === code ? styles.segmentActive : ""]
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

      <div className={styles.terms}>
        <button
          type="button"
          className={styles.termsToggle}
          onClick={() => setTermsOpen((open) => !open)}
          aria-expanded={termsOpen}
        >
          {termsOpen ? "▾" : "▸"} Custom terms (optional)
        </button>
        {termsOpen && (
          <>
            <p className={styles.hint}>
              Fix words the model mishears. Used for this file only, then discarded.
            </p>
            <textarea
              className={[styles.termsInput, termsError ? styles.termsInvalid : ""]
                .filter(Boolean)
                .join(" ")}
              rows={4}
              spellCheck={false}
              disabled={busy}
              value={termsText}
              onChange={(e) => onTermsChange(e.target.value)}
              placeholder={'{ "reyki": "reiki", "çakıra": "çakra" }'}
            />
            {termsError && <p className={styles.termsError}>{termsError}</p>}
          </>
        )}
      </div>

      {turnstileSiteKey && (
        <div className={styles.captcha}>
          <div ref={captchaRef} />
        </div>
      )}

      <div className={styles.actions}>
        <button type="button" className={styles.primary} onClick={start} disabled={!canStart}>
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
              className={[styles.progressBar, queued ? styles.progressIndeterminate : ""]
                .filter(Boolean)
                .join(" ")}
              style={queued ? undefined : { width: `${pct}%` }}
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
