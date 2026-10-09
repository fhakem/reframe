import { useRef, useState, type DragEvent } from "react";

interface Props {
  accept?: string[];
  maxMb?: number;
  disabled?: boolean;
  onFile: (file: File) => void;
}

export function Uploader({ accept, maxMb, disabled, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = (file: File | undefined) => {
    if (!file) return;
    const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
    if (accept && !accept.includes(ext)) {
      setError(`Unsupported file type ${ext}. Accepted: ${accept.join(", ")}`);
      return;
    }
    if (maxMb && file.size > maxMb * 1024 * 1024) {
      setError(`File is larger than ${maxMb} MB`);
      return;
    }
    setError(null);
    onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) pick(e.dataTransfer.files[0]);
  };

  return (
    <div>
      <div
        className={`dropzone${over ? " over" : ""}${disabled ? " disabled" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a video"
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <strong>Drop a 16:9 video here</strong>
        <span>or click to choose{maxMb ? ` · up to ${maxMb} MB` : ""}</span>
        <input
          ref={input}
          type="file"
          hidden
          accept={accept?.join(",") ?? "video/*"}
          onChange={(e) => {
            pick(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
