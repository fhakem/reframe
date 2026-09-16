/**
 * Turn the pydantic JSON schema of ReframeConfig into form field descriptors.
 *
 * Nothing here knows the config's fields: new fields appear automatically.
 * Known shapes get a dedicated widget; anything else falls back to a JSON box.
 */

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  pattern?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  multipleOf?: number;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  [k: string]: unknown;
}

export type FieldKind = "select" | "range" | "number" | "checkbox" | "text" | "json";

export interface Field {
  name: string;
  label: string;
  description?: string;
  kind: FieldKind;
  default: unknown;
  nullable: boolean;
  integer: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

export interface Preset {
  width: number;
  height: number;
}

export interface SchemaResponse {
  schema: JsonSchema;
  defaults: Record<string, unknown>;
  presets: Record<string, Preset>;
  accept?: string[];
  max_upload_mb?: number;
}

/** Fields shown first, in this order; everything else follows in schema order. */
export const PRIMARY_FIELDS = ["preset", "seam_ratio"];

function resolveRef(ref: string, root: JsonSchema): JsonSchema {
  if (!ref.startsWith("#/")) return {};
  let node: unknown = root;
  for (const part of ref.slice(2).split("/")) {
    if (node && typeof node === "object") node = (node as Record<string, unknown>)[part];
    else return {};
  }
  return (node as JsonSchema) ?? {};
}

/** Merge $ref / single allOf into the node; detect `anyOf: [X, {type: null}]`. */
export function normalize(node: JsonSchema, root: JsonSchema): { s: JsonSchema; nullable: boolean } {
  let s: JsonSchema = { ...node };
  let nullable = false;
  for (let guard = 0; guard < 8; guard++) {
    if (s.$ref) {
      const { $ref, ...rest } = s;
      s = { ...resolveRef($ref, root), ...rest };
      continue;
    }
    if (s.allOf && s.allOf.length === 1) {
      const { allOf, ...rest } = s;
      s = { ...allOf[0], ...rest };
      continue;
    }
    const alts = s.anyOf ?? s.oneOf;
    if (alts) {
      const nonNull = alts.filter((a) => a.type !== "null");
      if (nonNull.length < alts.length) nullable = true;
      if (nonNull.length === 1) {
        const { anyOf: _a, oneOf: _o, ...rest } = s;
        s = { ...nonNull[0], ...rest };
        continue;
      }
    }
    if (Array.isArray(s.type)) {
      const types = s.type.filter((t) => t !== "null");
      if (types.length < s.type.length) nullable = true;
      s = { ...s, type: types.length === 1 ? types[0] : types };
    }
    break;
  }
  return { s, nullable };
}

/** `^(a|b|c)$` -> ["a", "b", "c"] (pydantic `pattern=` used as an enum). */
export function patternOptions(pattern: string | undefined): string[] | undefined {
  const m = pattern?.match(/^\^\(([\w.-]+(?:\|[\w.-]+)+)\)\$$/);
  return m ? m[1].split("|") : undefined;
}

function niceStep(span: number, integer: boolean): number {
  if (integer) return 1;
  // Smallest power of ten giving at most ~100 slider positions (0.75 -> 0.01).
  const p = Math.ceil(Math.log10(span / 100) - 1e-9);
  return Number(Math.pow(10, p).toPrecision(1));
}

function humanize(name: string): string {
  const s = name.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function fieldFromSchema(name: string, node: JsonSchema, root: JsonSchema): Field {
  const { s, nullable } = normalize(node, root);
  const base = {
    name,
    label: node.title ?? humanize(name),
    description: s.description,
    default: s.default ?? node.default,
    nullable,
    integer: s.type === "integer",
  };
  if (s.enum && s.enum.every((e) => typeof e === "string" || typeof e === "number")) {
    return { ...base, kind: "select", options: s.enum.map(String) };
  }
  const pat = s.type === "string" ? patternOptions(s.pattern) : undefined;
  if (pat) return { ...base, kind: "select", options: pat };
  if (s.type === "boolean") return { ...base, kind: "checkbox" };
  if (s.type === "integer" || s.type === "number") {
    const integer = s.type === "integer";
    const lo = s.minimum ?? s.exclusiveMinimum;
    const hi = s.maximum ?? s.exclusiveMaximum;
    if (lo !== undefined && hi !== undefined && hi > lo) {
      const step = s.multipleOf ?? niceStep(hi - lo, integer);
      const min = s.minimum ?? lo + step;
      const max = s.maximum ?? hi - step;
      return { ...base, kind: "range", min, max, step };
    }
    return { ...base, kind: "number", min: lo, max: hi, step: integer ? 1 : undefined };
  }
  if (s.type === "string") return { ...base, kind: "text" };
  return { ...base, kind: "json" };
}

export function fieldsFromSchema(schema: JsonSchema): Field[] {
  const props = schema.properties ?? {};
  const names = Object.keys(props);
  const ordered = [
    ...PRIMARY_FIELDS.filter((n) => names.includes(n)),
    ...names.filter((n) => !PRIMARY_FIELDS.includes(n)),
  ];
  return ordered.map((n) => fieldFromSchema(n, props[n], schema));
}

/** Parse a raw input string into the value to send for ``field``. */
export function coerce(field: Field, raw: string | boolean): unknown {
  if (field.kind === "checkbox") return Boolean(raw);
  const str = String(raw);
  if (str === "" && field.nullable) return null;
  if (field.kind === "range" || field.kind === "number") {
    if (str.trim() === "") return field.nullable ? null : undefined;
    const n = Number(str);
    if (Number.isNaN(n)) return undefined;
    return field.integer ? Math.round(n) : n;
  }
  if (field.kind === "json") {
    try {
      return JSON.parse(str);
    } catch {
      return undefined;
    }
  }
  return str;
}
