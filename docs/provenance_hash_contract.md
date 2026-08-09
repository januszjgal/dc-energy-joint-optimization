# Provenance hash contract

Contract ID: `dc-energy-provenance-sha256-v2`

This contract separates content identity from working-tree representation:

- Tracked source and other non-JSON text are identified by SHA-256 of the exact
  Git blob bytes at an explicit commit and path. Verification reads the object
  database, not the checkout.
- JSON evidence is parsed and serialized as UTF-8 JSON with sorted object keys,
  separators `,` and `:`, no insignificant whitespace, no trailing newline,
  non-ASCII characters emitted as UTF-8, and non-finite numbers rejected.
  SHA-256 is computed over those canonical bytes.
- Binary and data artifacts are identified by SHA-256 of their exact raw bytes.
  No text or EOL normalization is applied.
- Internal model policy and critic identities remain SHA-256 digests over the
  ordered tensor-state bytes defined by `ramp_rl.runner.model_hashes`.
- A bundle digest is canonical JSON SHA-256 over the declared commit and the
  sorted path-to-hash map. It is not a concatenation of checkout-byte hashes.

`.gitattributes` enforces LF for tracked source/evidence text and marks raw data
and binary formats as non-text. Correctness does not depend on those checkout
settings: Git-blob and canonical-JSON verification is stable under either
`core.autocrlf=true` or `core.autocrlf=false`.

Legacy V4R hashes are retained only as migration evidence. The resealed package
classifies each old digest by the representation that produced it and uses this
contract for every new edge. It supersedes provenance bindings only; it does
not supersede the protocol, controller, model, data, chronology, or results.
