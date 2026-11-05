// SPDX-License-Identifier: Apache-2.0
//
// Raygen-side hit ordering.
//
// The any-hit maintains the k-nearest hits UNSORTED (moving the O(k) ordering
// work off the traversal critical path). The raygen then calls
// loadAndSortHits() once per window: it copies the payload ONCE into thread-local
// arrays and sorts those with a Batcher odd-even merge sort. All compare-exchange
// indices are compile-time constants after unrolling (no data-dependent indexing),
// so the locals can live in registers -- the payload is read exactly once per
// window instead of being re-traversed by every sort compare/swap and every
// integrate-loop access.
//
// Unwritten slots hold INF_T, so the full-array sort pushes them to the tail.
//
// #include AFTER the `rayPayloadEXT HitPayload prd;` declaration.

// oid[] carries buffer SLOT indices (0..k-1); the raygen resolves the gaussian id
// per hit through the per-ray scratch (HIT_ID macro).
void loadAndSortHits(out uint oid[MAX_HITS_PER_TRACE], out float ot[MAX_HITS_PER_TRACE]) {
    // single pass over the payload
    for (int i = 0; i < MAX_HITS_PER_TRACE; ++i) {
        oid[i] = uint(i);
        ot[i]  = prd.hdt[i];
    }

    // Batcher odd-even merge sort for arbitrary n. All loop bounds derive from
    // the compile-time constant MAX_HITS_PER_TRACE, allowing the compiler to
    // unroll this into a fixed compare-exchange network with constant indices.
    const int n = MAX_HITS_PER_TRACE;
    for (int p = 1; p < n; p <<= 1) {
        for (int q = p; q >= 1; q >>= 1) {
            for (int j = q % p; j <= n - 1 - q; j += 2 * q) {
                int iMax = min(q - 1, n - j - q - 1);
                for (int i = 0; i <= iMax; ++i) {
                    if ((i + j) / (2 * p) == (i + j + q) / (2 * p)) {
                        int a = i + j, b = i + j + q;
                        if (ot[b] < ot[a]) {
                            float tt = ot[a];  ot[a]  = ot[b];  ot[b]  = tt;
                            uint  ti = oid[a]; oid[a] = oid[b]; oid[b] = ti;
                        }
                    }
                }
            }
        }
    }
}
