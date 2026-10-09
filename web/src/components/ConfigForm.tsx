import { useMemo, useState } from "react";
import { coerce, fieldsFromSchema, PRIMARY_FIELDS, type Field, type SchemaResponse } from "../schema";

export type ConfigValue = Record<string, unknown>;

interface Props {
  schema: SchemaResponse;
  value: ConfigValue;
  onChange: (next: ConfigValue) => void;
  disabled?: boolean;
}

function JsonInput({
  field,
  value,
  onValue,
  disabled,
}: {
  field: Field;
  value: unknown;
  onValue: (v: unknown) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState(() => JSON.stringify(value ?? null, null, 0));
  const [bad, setBad] = useState(false);
  return (
    <textarea
      id={`cfg-${field.name}`}
      rows={2}
      spellCheck={false}
      aria-invalid={bad}
      disabled={disabled}
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        const v = coerce(field, e.target.value);
        setBad(v === undefined);
        if (v !== undefined) onValue(v);
      }}
    />
  );
}

/** Keeps a local draft so the field can be cleared / half-typed; commits only valid numbers. */
function NumberInput({
  field,
  value,
  onValue,
  disabled,
}: {
  field: Field;
  value: unknown;
  onValue: (v: unknown) => void;
  disabled?: boolean;
}) {
  const external = value == null ? "" : String(value);
  const [draft, setDraft] = useState(external);
  const [synced, setSynced] = useState(external);
  if (external !== synced) {
    // Parent changed the value (e.g. reset): adopt it.
    setSynced(external);
    setDraft(external);
  }
  const parsed = coerce(field, draft);
  const outOfRange =
    typeof parsed === "number" &&
    ((field.min !== undefined && parsed < field.min) ||
      (field.max !== undefined && parsed > field.max));
  return (
    <input
      id={`cfg-${field.name}`}
      type="number"
      inputMode={field.integer ? "numeric" : "decimal"}
      min={field.min}
      max={field.max}
      step={field.step ?? "any"}
      disabled={disabled}
      aria-invalid={parsed === undefined || outOfRange}
      value={draft}
      onChange={(e) => {
        const raw = e.target.value;
        setDraft(raw);
        const v = coerce(field, raw);
        if (v === undefined) return;
        const bad =
          typeof v === "number" &&
          ((field.min !== undefined && v < field.min) ||
            (field.max !== undefined && v > field.max));
        if (!bad) {
          setSynced(v == null ? "" : String(v));
          onValue(v);
        }
      }}
    />
  );
}

function FieldInput({
  field,
  value,
  onValue,
  schema,
  disabled,
}: {
  field: Field;
  value: unknown;
  onValue: (v: unknown) => void;
  schema: SchemaResponse;
  disabled?: boolean;
}) {
  const id = `cfg-${field.name}`;
  switch (field.kind) {
    case "select":
      return (
        <select
          id={id}
          disabled={disabled}
          value={value == null ? "" : String(value)}
          onChange={(e) => onValue(coerce(field, e.target.value))}
        >
          {field.nullable && <option value="">—</option>}
          {field.options!.map((o) => {
            const p = field.name === "preset" ? schema.presets[o] : undefined;
            return (
              <option key={o} value={o}>
                {p ? `${o} (${p.width}×${p.height})` : o}
              </option>
            );
          })}
        </select>
      );
    case "checkbox":
      return (
        <input
          id={id}
          type="checkbox"
          disabled={disabled}
          checked={Boolean(value)}
          onChange={(e) => onValue(e.target.checked)}
        />
      );
    case "range": {
      const num = typeof value === "number" ? value : Number(field.default ?? field.min);
      const isSeam = field.name === "seam_ratio";
      return (
        <div className="range">
          {isSeam && <span className="range-end">crop</span>}
          <input
            id={id}
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            disabled={disabled}
            value={num}
            onChange={(e) => {
              const v = coerce(field, e.target.value);
              if (v !== undefined) onValue(v);
            }}
          />
          {isSeam && <span className="range-end">seams</span>}
          <output htmlFor={id} className="range-value">
            {isSeam ? `${Math.round(num * 100)}%` : num}
          </output>
        </div>
      );
    }
    case "number":
      return <NumberInput field={field} value={value} onValue={onValue} disabled={disabled} />;
    case "text":
      return (
        <input
          id={id}
          type="text"
          disabled={disabled}
          value={value == null ? "" : String(value)}
          onChange={(e) => onValue(coerce(field, e.target.value))}
        />
      );
    default:
      return <JsonInput field={field} value={value} onValue={onValue} disabled={disabled} />;
  }
}

/** Form generated from the ReframeConfig JSON schema served by the API. */
export function ConfigForm({ schema, value, onChange, disabled }: Props) {
  const fields = useMemo(() => fieldsFromSchema(schema.schema), [schema]);
  const primary = fields.filter((f) => PRIMARY_FIELDS.includes(f.name));
  const advanced = fields.filter((f) => !PRIMARY_FIELDS.includes(f.name));

  const render = (f: Field) => (
    <div className={`field field-${f.kind}`} key={f.name}>
      <label htmlFor={`cfg-${f.name}`} title={f.description}>
        {f.name === "seam_ratio" ? "Crop ↔ seams" : f.label}
      </label>
      <FieldInput
        field={f}
        schema={schema}
        disabled={disabled}
        value={f.name in value ? value[f.name] : f.default}
        onValue={(v) => onChange({ ...value, [f.name]: v })}
      />
      {f.description && f.name === "seam_ratio" && (
        <small className="hint">
          Left: pure centre crop. Right: more of the frame kept, squeezed by seam carving (above
          ~35% seams start cutting through subjects).
        </small>
      )}
    </div>
  );

  return (
    <form className="config-form" onSubmit={(e) => e.preventDefault()} aria-label="Settings">
      {primary.map(render)}
      {advanced.length > 0 && (
        <details className="advanced">
          <summary>Advanced</summary>
          {advanced.map(render)}
          <button
            type="button"
            className="link"
            disabled={disabled}
            onClick={() => onChange({ ...schema.defaults })}
          >
            Reset to defaults
          </button>
        </details>
      )}
    </form>
  );
}
