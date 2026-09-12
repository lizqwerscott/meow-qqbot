import { describe, expect, it } from "vitest";
import {
  calculateVirtualRange,
  isFollowingTail,
  mergeUniqueSessions,
  restoreScrollTop,
  VirtualHeightIndex,
} from "./transcript-state";

describe("transcript state helpers", () => {
  it("merges session pages without duplicating sessions", () => {
    const result = mergeUniqueSessions(
      [{ session_id: "s-1" }, { session_id: "s-2" }],
      [{ session_id: "s-2" }, { session_id: "s-3" }, { session_id: "s-3" }],
    );

    expect(result.map((session) => session.session_id)).toEqual([
      "s-1",
      "s-2",
      "s-3",
    ]);
  });

  it("detects whether the transcript is following its tail", () => {
    expect(isFollowingTail(1_000, 700, 280)).toBe(true);
    expect(isFollowingTail(1_000, 600, 280)).toBe(false);
  });

  it("restores the relative scroll position after prepend", () => {
    expect(restoreScrollTop(1_000, 240, 1_640)).toBe(880);
  });

  it("keeps short transcripts out of virtualization", () => {
    expect(
      calculateVirtualRange(["a", "b"], 0, 400, 180, () => undefined),
    ).toEqual({ virtual: false, start: 0, end: 2, top: 0, bottom: 0 });
  });

  it("calculates a window and measured spacer heights for long transcripts", () => {
    const ids = Array.from({ length: 200 }, (_, index) => `turn-${index}`);
    const range = calculateVirtualRange(
      ids,
      1_800,
      360,
      180,
      (id) => (id === "turn-0" ? 240 : undefined),
      120,
      2,
    );

    expect(range.virtual).toBe(true);
    expect(range.start).toBe(8);
    expect(range.end).toBe(14);
    expect(range.top).toBe(1_500);
    expect(range.bottom).toBe((200 - 14) * 180);
  });

  it("keeps indexed spacer queries fast for a 10,000-turn transcript", () => {
    const ids = Array.from({ length: 10_000 }, (_, index) => `turn-${index}`);
    const heightIndex = new VirtualHeightIndex();
    heightIndex.sync(ids, 180, (id) => (id === "turn-0" ? 240 : undefined));

    expect(heightIndex.prefix(8)).toBe(1_500);
    expect(heightIndex.total()).toBe(10_000 * 180 + 60);

    heightIndex.setMeasuredHeight("turn-5000", 420);
    expect(heightIndex.total()).toBe(10_000 * 180 + 300);
    expect(heightIndex.prefix(5_001)).toBe(5_001 * 180 + 60 + 240);
  });

  it("uses indexed heights for virtual spacer calculations", () => {
    const ids = Array.from({ length: 200 }, (_, index) => `turn-${index}`);
    const heightIndex = new VirtualHeightIndex();
    const range = calculateVirtualRange(
      ids,
      1_800,
      360,
      180,
      (id) => (id === "turn-0" ? 240 : undefined),
      120,
      2,
      heightIndex,
    );

    expect(range.top).toBe(1_500);
    expect(range.bottom).toBe((200 - 14) * 180);
  });
});
