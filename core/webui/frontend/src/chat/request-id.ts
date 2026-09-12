let fallbackSequence = 0;

function formatUuid(bytes: Uint8Array): string {
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function createRequestId(prefix: string, cryptoApi: Crypto | undefined = globalThis.crypto): string {
  if (typeof cryptoApi?.randomUUID === "function") {
    try {
      return `${prefix}-${cryptoApi.randomUUID()}`;
    } catch {
    }
  }

  if (typeof cryptoApi?.getRandomValues === "function") {
    try {
      return `${prefix}-${formatUuid(cryptoApi.getRandomValues(new Uint8Array(16)))}`;
    } catch {
    }
  }

  fallbackSequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${fallbackSequence.toString(36)}-${Math.random().toString(36).slice(2)}`;
}
