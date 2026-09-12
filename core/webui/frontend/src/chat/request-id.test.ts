import { describe, expect, it } from "vitest";
import { createRequestId } from "./request-id";

describe("createRequestId", () => {
  it("falls back when randomUUID is unavailable", () => {
    const fallbackCrypto = {
      getRandomValues(bytes: Uint8Array) {
        bytes.fill(1);
        return bytes;
      },
    } as unknown as Crypto;

    expect(createRequestId("webui", fallbackCrypto)).toMatch(
      /^webui-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("works without a crypto implementation", () => {
    expect(createRequestId("webui", undefined)).toMatch(/^webui-[a-z0-9-]+$/);
  });
});
