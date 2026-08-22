# Architecture

The canonical import namespace is `beyond_backprop`. Configuration, runtime
policy, dataset loading, architecture construction, algorithm adapters, and the
training runner live under `src/beyond_backprop/`.

The runner owns the shared lifecycle: validate the typed configuration, resolve
the backend and device, seed the process, prepare data, build the model and
adapter, create optional services, train, restore best state, evaluate, and
persist artifacts. BP, FF, CaFo, and MF adapters retain their own lifecycle
state and delegate legacy mathematical implementations while extraction is in
progress.

Modules under the historical `src.*` namespace remain compatibility entry
points. New implementation code should import the canonical namespace.
