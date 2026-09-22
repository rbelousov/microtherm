"""Shared spawn-safe simulation workers and notebook progress display.

Keep workers in this importable module: spawned processes cannot import
functions defined in a notebook's __main__ namespace.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from queue import Empty
from time import perf_counter

import numpy as np
from scipy.special import rel_entr
from microtherm import BoltzmannGas, Model


def local_equilibrium_kl(model):
    levels = model.state.levels
    occupancies = model.state.site_occupancies()
    occupied = occupancies > 0
    kl = np.full(len(occupancies), np.nan)
    conditional = levels[occupied] / occupancies[occupied, None]
    equilibrium = model.local_equilibrium()[occupied]
    kl[occupied] = rel_entr(conditional, equilibrium).sum(axis=1)
    return kl


def validate_state(model, require_transition=True):
    nsites, nlevels = model.state.levels.shape
    state = model.state
    assert np.all(np.isfinite(state.levels))
    assert np.all(state.levels >= 0)
    assert np.all(np.isfinite(state.betas))
    np.testing.assert_allclose(state.levels.sum(), 1.0, atol=1e-12, rtol=0)
    if require_transition:
        transition = state.transition
        assert transition.shape == (nsites, nlevels, nsites, nlevels)
        assert np.all(np.isfinite(transition)) and np.all(transition >= 0)
        np.testing.assert_allclose(transition.sum(axis=(0, 1)), 1.0, atol=1e-12, rtol=0)
        for site, beta in model.fixed.items():
            assert state.betas[site] == beta


def run_case(
    name,
    bath_betas,
    nsites,
    nlevels,
    n_steps,
    report_every,
    model_options=None,
    progress=None,
):
    """Run one independent case; optionally send progress to the parent."""
    if n_steps < 1 or report_every < 1:
        raise ValueError("n_steps and report_every must be positive")
    beta_left, beta_right = bath_betas
    initial = BoltzmannGas(levels=np.full((nsites, nlevels), 1.0 / (nsites * nlevels)))
    model = Model(
        state=initial,
        **(model_options or {}),
        fixed_betas={0: beta_left, nsites - 1: beta_right},
    )
    validate_state(model, require_transition=False)
    history = np.empty((n_steps + 1, nsites))
    history[0] = local_equilibrium_kl(model)
    started = perf_counter()
    for step in range(1, n_steps + 1):
        previous = model.state
        model()
        model.state.meta["frame"] = step
        validate_state(model)
        history[step] = local_equilibrium_kl(model)
        if progress is not None:
            diagnostics = None
            if step % report_every == 0 or step == n_steps:
                diagnostics = {
                    "mass": f"{model.state.levels.sum():.12f}",
                    "KL min": f"{np.nanmin(history[step]):.3e}",
                    "KL max": f"{np.nanmax(history[step]):.3e}",
                }
            progress.put((name, step, diagnostics))

    delta_p = np.max(np.abs(model.state.levels - previous.levels))
    delta_beta = np.max(np.abs(model.state.betas - previous.betas))
    assert history.shape == (n_steps + 1, nsites)
    assert np.all(np.isfinite(history))
    assert np.all(history >= -1e-12)
    return {
        "model": model,
        "kl": history,
        "delta_p": delta_p,
        "delta_beta": delta_beta,
        "elapsed": perf_counter() - started,
    }


def run_cases(bath_betas, nsites, nlevels, n_steps, report_every, model_options=None):
    """Run cases concurrently, updating all bars in the calling output cell.

    Results retain input order. Worker exceptions propagate to the notebook;
    failure or interruption terminates the remaining workers (Python 3.14+).
    """
    from tqdm.auto import tqdm

    if n_steps < 1 or report_every < 1:
        raise ValueError("n_steps and report_every must be positive")
    if not bath_betas:
        return {}
    context = mp.get_context("spawn")
    bars = {}
    with context.Manager() as manager:
        progress = manager.Queue()
        executor = ProcessPoolExecutor(
            max_workers=len(bath_betas),
            mp_context=context,
        )
        try:
            for position, name in enumerate(bath_betas):
                bars[name] = tqdm(
                    total=n_steps,
                    desc=name,
                    position=position,
                    mininterval=0.2,
                    leave=True,
                )
            futures = {
                name: executor.submit(
                    run_case,
                    name,
                    betas,
                    nsites,
                    nlevels,
                    n_steps,
                    report_every,
                    model_options,
                    progress,
                )
                for name, betas in bath_betas.items()
            }

            def update(message):
                name, step, diagnostics = message
                bar = bars[name]
                if diagnostics is not None:
                    bar.set_postfix(diagnostics, refresh=False)
                bar.update(step - bar.n)

            while True:
                # Check futures even if a worker fails before sending progress.
                for future in futures.values():
                    if future.done():
                        future.result()
                if all(future.done() for future in futures.values()):
                    break
                try:
                    update(progress.get(timeout=0.1))
                except Empty:
                    pass
            # Queue puts finish before the corresponding future completes.
            while True:
                try:
                    update(progress.get_nowait())
                except Empty:
                    break
            results = {name: future.result() for name, future in futures.items()}
        except BaseException:
            executor.terminate_workers()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for bar in bars.values():
                bar.close()

    for name, result in results.items():
        print(
            f"{name}: max |delta p|={result['delta_p']:.3e}; "
            f"max |delta beta|={result['delta_beta']:.3e}; "
            f"elapsed={result['elapsed']:.1f} s"
        )
    return results
