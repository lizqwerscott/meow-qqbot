export type SessionLike = { session_id: string };

export type VirtualRange = {
  virtual: boolean;
  start: number;
  end: number;
  top: number;
  bottom: number;
};

export class VirtualHeightIndex {
  private ids: string[] = [];
  private positions = new Map<string, number>();
  private values: number[] = [];
  private tree: number[] = [];
  private estimate = 180;

  sync(
    itemIds: readonly string[],
    estimatedHeight: number,
    measuredHeight: (itemId: string) => number | undefined,
  ) {
    const safeEstimate = Math.max(1, estimatedHeight);
    if (this.estimate === safeEstimate && this.sameIds(itemIds)) return;
    this.ids = [...itemIds];
    this.positions = new Map(this.ids.map((itemId, index) => [itemId, index]));
    this.estimate = safeEstimate;
    this.values = this.ids.map((itemId) => this.heightOf(itemId, measuredHeight));
    this.tree = Array(this.values.length + 1).fill(0);
    this.values.forEach((value, index) => this.updateTree(index, value));
  }

  setMeasuredHeight(itemId: string, measuredHeight: number) {
    const index = this.positions.get(itemId);
    if (index === undefined || !Number.isFinite(measuredHeight) || measuredHeight <= 0) return;
    const nextHeight = Math.max(1, measuredHeight);
    const previousHeight = this.values[index];
    if (Math.abs(previousHeight - nextHeight) < 1) return;
    this.values[index] = nextHeight;
    this.updateTree(index, nextHeight - previousHeight);
  }

  prefix(count: number) {
    let index = Math.min(Math.max(0, count), this.values.length);
    let total = 0;
    while (index > 0) {
      total += this.tree[index];
      index -= index & -index;
    }
    return total;
  }

  total() {
    return this.prefix(this.values.length);
  }

  private sameIds(itemIds: readonly string[]) {
    return itemIds.length === this.ids.length && itemIds.every((itemId, index) => itemId === this.ids[index]);
  }

  private heightOf(itemId: string, measuredHeight: (itemId: string) => number | undefined) {
    const value = measuredHeight(itemId);
    return typeof value === "number" && Number.isFinite(value) && value > 0
      ? value
      : this.estimate;
  }

  private updateTree(index: number, delta: number) {
    for (let cursor = index + 1; cursor < this.tree.length; cursor += cursor & -cursor) {
      this.tree[cursor] += delta;
    }
  }
}

export function mergeUniqueSessions<T extends SessionLike>(
  existing: readonly T[],
  incoming: readonly T[],
): T[] {
  const known = new Set(existing.map((session) => session.session_id));
  const merged = [...existing];
  for (const session of incoming) {
    if (known.has(session.session_id)) continue;
    known.add(session.session_id);
    merged.push(session);
  }
  return merged;
}

export function isFollowingTail(
  scrollHeight: number,
  scrollTop: number,
  clientHeight: number,
  threshold = 80,
): boolean {
  return scrollHeight - scrollTop - clientHeight < threshold;
}

export function restoreScrollTop(
  previousScrollHeight: number,
  previousScrollTop: number,
  nextScrollHeight: number,
): number {
  return nextScrollHeight - previousScrollHeight + previousScrollTop;
}

export function calculateVirtualRange(
  itemIds: readonly string[],
  viewportTop: number,
  viewportHeight: number,
  estimatedHeight: number,
  measuredHeight: (itemId: string) => number | undefined,
  threshold = 120,
  overscan = 8,
  heightIndex?: VirtualHeightIndex,
): VirtualRange {
  const virtual = itemIds.length > threshold;
  if (!virtual) {
    return { virtual: false, start: 0, end: itemIds.length, top: 0, bottom: 0 };
  }
  const safeEstimate = Math.max(1, estimatedHeight);
  heightIndex?.sync(itemIds, safeEstimate, measuredHeight);
  const start = Math.max(
    0,
    Math.floor(viewportTop / safeEstimate) - overscan,
  );
  const visibleCount = Math.ceil(viewportHeight / safeEstimate) + overscan * 2;
  const end = Math.min(itemIds.length, start + visibleCount);
  if (heightIndex) {
    return {
      virtual: true,
      start,
      end,
      top: heightIndex.prefix(start),
      bottom: heightIndex.total() - heightIndex.prefix(end),
    };
  }
  const heightOf = (itemId: string) => measuredHeight(itemId) || safeEstimate;
  const top = itemIds
    .slice(0, start)
    .reduce((total, itemId) => total + heightOf(itemId), 0);
  const bottom = itemIds
    .slice(end)
    .reduce((total, itemId) => total + heightOf(itemId), 0);
  return { virtual: true, start, end, top, bottom };
}
