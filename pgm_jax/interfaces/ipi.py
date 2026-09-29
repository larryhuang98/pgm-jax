"""i-PI socket client (driver) for pGM: i-PI integrates (PIMD, TRPMD, thermostats, barostats,
enhanced sampling), pgm_jax computes energies, forces and virials.

Pure Python sockets; i-PI itself is only needed on the server side.  The client speaks the i-PI
protocol (12-byte headers STATUS / INIT / POSDATA / GETFORCE / EXIT; atomic units: Bohr, Hartree),
including i-PI's batched requests (INIT string "batch_size:n": one POSDATA carries n structures,
e.g. the beads of a ring polymer, answered in one GETFORCE).  A batch with one cell is evaluated in
one vmapped call (PGMEngine.compute_batch; vmap_beads=False: one call per structure).  Each
structure keeps its own induced-dipole history: i-PI does not keep the order of the beads within
its batches, so each structure is matched to the closest previous one (one-to-one assignment).
Without batching the engine picks the slot whose last configuration is closest (PGMEngine(slots=P)
for P beads sent one after the other).

    python -m pgm_jax.interfaces.ipi --prmtop water.prmtop [--template water.flex] \
        --address pgm --unix [--slots 32] [--precision mixed] [--settings '{"dipole_tol": 1e-5}']

or from Python:

    client = IPIClient(lambda pos, H: PGMEngine(sys, pos, H, settings, templates=...), "pgm", unix=True)
    client.run()                               # until i-PI sends EXIT

The engine is built on the first structure (i-PI's positions and cell).  Atoms must be in the
system's order.  Virial: i-PI expects sum_i r_i (x) f_i = -dE/d eps: the engine's atomic virial for flexible templates
(the default there).  Extras (JSON; i-PI's `dipole` property and <extras> output): the cell dipole M_q + M_perm + M_ind
(e Bohr) and the CG iterations of the dipole solve."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import numpy as np

BOHR_NM = 0.0529177210544  # nm per Bohr (CODATA 2022)
HARTREE_KJMOL = 2625.4996394799  # kJ/mol per Hartree
HDRLEN = 12


def _msg(s: str) -> bytes:
    return s.upper().ljust(HDRLEN).encode()


STATUS, INIT, POSDATA, GETFORCE, EXIT = (_msg(x) for x in ("STATUS", "INIT", "POSDATA", "GETFORCE", "EXIT"))
READY, HAVEDATA, NEEDINIT, FORCEREADY = (_msg(x) for x in ("READY", "HAVEDATA", "NEEDINIT", "FORCEREADY"))


class IPIClient:
    """i-PI driver around a PGMEngine (or any object with the engine's compute()).

    engine: a PGMEngine, or a factory f(pos_nm, cell_nm) -> engine called on the first structure.
    address: host name (inet) or socket name (unix: sockets_prefix + address); port for inet."""

    def __init__(
        self,
        engine,
        address: str = "localhost",
        port: int = 31415,
        unix: bool = False,
        sockets_prefix: str = "/tmp/ipi_",
        virial: bool = True,
        verbose: bool = False,
        log=sys.stdout,
        vmap_beads: bool = True,
        dipole: bool = True,
    ):
        self._engine = None if callable(engine) and not hasattr(engine, "compute") else engine
        self._factory = engine if self._engine is None else None
        self.address, self.port, self.unix, self.prefix = address, int(port), bool(unix), sockets_prefix
        self.virial, self.verbose, self.log = bool(virial), verbose, log
        self.vmap_beads = bool(vmap_beads)  # batches of structures with one cell: one vmapped call
        self.dipole = bool(dipole)  # cell dipole in the extras of every structure
        self.batch = 1
        self.stats = {"structures": 0, "requests": 0, "t_engine": 0.0, "t_total": 0.0}

    @property
    def engine(self):
        return self._engine

    # ------------------------------------------------------------------ socket helpers
    def _connect(self, retries: int = 600, wait: float = 0.5):
        for _k in range(retries):
            try:
                if self.unix:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.prefix + self.address)
                else:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.connect((self.address, self.port))
                return s
            except (ConnectionRefusedError, FileNotFoundError):
                s.close()
                time.sleep(wait)
        raise ConnectionError(f"cannot connect to i-PI at {self.address}")

    def _recv(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("i-PI closed the connection")
            got += k
        return bytes(buf)

    def _recv_array(self, dtype, count: int):
        dt = np.dtype(dtype)
        return np.frombuffer(self._recv(dt.itemsize * count), dt)

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, h, pos_bohr, slot: int | None):
        """One structure: h (3, 3) i-PI cell (lattice vectors as columns, Bohr), positions (Bohr).
        Returns (energy Ha, forces Ha/Bohr (N, 3), virial Ha (3, 3), extras dict)."""
        cell = np.asarray(h, float).T * BOHR_NM  # rows = lattice vectors, nm
        pos = np.asarray(pos_bohr, float).reshape(-1, 3) * BOHR_NM
        self._ensure(h, pos_bohr)
        t0 = time.perf_counter()
        eng = self._engine
        if slot is not None and hasattr(eng, "slots") and len(eng.slots) > 1:
            res = eng.compute(pos, cell, virial=self.virial, slot=slot % len(eng.slots))
        else:
            res = eng.compute(pos, cell, virial=self.virial)
        self.stats["t_engine"] += time.perf_counter() - t0
        self.stats["structures"] += 1
        return self._convert(res)

    def _ensure(self, h, pos_bohr):
        """Build the engine from the factory on the first structure."""
        if self._engine is None:
            cell = np.asarray(h, float).T * BOHR_NM
            self._engine = self._factory(np.asarray(pos_bohr, float).reshape(-1, 3) * BOHR_NM, cell)
            self._print(f"# {self._engine.describe()}" if hasattr(self._engine, "describe") else "# engine ready")
        if self.dipole and hasattr(self._engine, "with_dipole"):
            self._engine.with_dipole = True  # the cell dipole in the same call (extras every step)

    def _convert(self, res):
        E = res.energy / HARTREE_KJMOL
        F = res.forces * (BOHR_NM / HARTREE_KJMOL)
        W = res.virial if (self.virial and res.virial is not None) else np.zeros((3, 3))
        vir = -0.5 * (W + W.T) / HARTREE_KJMOL
        extras = {"cg_iterations": int(res.iterations)}
        if self.dipole:
            extras["dipole"] = (res.dipole / BOHR_NM).tolist()
        return E, np.ascontiguousarray(F, np.float64), np.ascontiguousarray(vir, np.float64), extras

    def _evaluate_batch(self, cells, pos_bohr):
        """A batch of structures (i-PI batch_size > 1): one vmapped engine call when they share the
        cell (ring-polymer beads; PGMEngine.compute_batch), else one call per structure."""
        same_cell = all(np.array_equal(cells[0], c) for c in cells[1:])
        if self._engine is None or not hasattr(self._engine, "compute_batch") or not self.vmap_beads or not same_cell:
            slots = list(range(len(cells)))
            if same_cell and hasattr(self._engine, "batch_slots"):  # i-PI does not keep the order of the beads
                slots = self._engine.batch_slots(
                    np.asarray(pos_bohr, float).reshape(len(cells), -1, 3) * BOHR_NM,
                    np.asarray(cells[0], float).T * BOHR_NM,
                )
            return [self._evaluate(cells[i], pos_bohr[i], int(slots[i])) for i in range(len(cells))]
        t0 = time.perf_counter()
        cell = np.asarray(cells[0], float).T * BOHR_NM
        res = self._engine.compute_batch(np.asarray(pos_bohr, float) * BOHR_NM, cell, virial=self.virial)
        self.stats["t_engine"] += time.perf_counter() - t0
        self.stats["structures"] += len(res)
        return [self._convert(r) for r in res]

    # ------------------------------------------------------------------ protocol
    def run(self, max_requests: int | None = None):
        """Serve i-PI until EXIT (or the connection closes, or max_requests POSDATA messages)."""
        self.sock = self._connect()
        initialised, have = False, False
        results = None
        t_start = time.perf_counter()
        try:
            while True:
                hdr = self._recv(HDRLEN)
                if hdr == STATUS:
                    self.sock.sendall(NEEDINIT if not initialised else (HAVEDATA if have else READY))
                elif hdr == INIT:
                    rid = int(self._recv_array(np.int32, 1)[0])
                    n = int(self._recv_array(np.int32, 1)[0])
                    text = self._recv(n).decode("utf-8", errors="replace")
                    for tok in text.split(","):
                        k, sep, v = tok.partition(":")
                        if sep and k.strip() == "batch_size":
                            self.batch = int(v.strip())
                    if self.verbose:
                        self._print(f"# INIT rid {rid} '{text}' batch {self.batch}")
                    initialised = True
                elif hdr == POSDATA:
                    if self.batch > 1:
                        nat = int(self._recv_array(np.int32, 1)[0])
                        cells = self._recv_array(np.float64, 18 * self.batch).reshape(self.batch, 2, 3, 3)
                        pos = self._recv_array(np.float64, 3 * nat * self.batch).reshape(self.batch, nat, 3)
                        self._ensure(cells[0, 0], pos[0])
                        results = self._evaluate_batch(cells[:, 0], pos)
                    else:
                        h = self._recv_array(np.float64, 9).reshape(3, 3)
                        self._recv_array(np.float64, 9)  # inverse cell (unused)
                        nat = int(self._recv_array(np.int32, 1)[0])
                        pos = self._recv_array(np.float64, 3 * nat)
                        results = [self._evaluate(h, pos, None)]
                    self.stats["requests"] += 1
                    have = True
                elif hdr == GETFORCE:
                    self.sock.sendall(FORCEREADY)
                    nat = results[0][1].shape[0]
                    if self.batch > 1:
                        payload = [
                            np.array([r[0] for r in results], np.float64).tobytes(),
                            np.int32(nat).tobytes(),
                            np.concatenate([r[1].reshape(-1) for r in results]).tobytes(),
                            np.concatenate([r[2].reshape(-1) for r in results]).tobytes(),
                        ]
                        for r in results:
                            x = json.dumps(r[3]).encode()
                            payload += [np.int32(len(x)).tobytes(), x]
                    else:
                        E, F, vir, ex = results[0]
                        x = json.dumps(ex).encode()
                        payload = [
                            np.float64(E).tobytes(),
                            np.int32(nat).tobytes(),
                            F.tobytes(),
                            vir.tobytes(),
                            np.int32(len(x)).tobytes(),
                            x,
                        ]
                    self.sock.sendall(b"".join(payload))
                    have = False
                    if max_requests is not None and self.stats["requests"] >= max_requests:
                        break
                elif hdr == EXIT:
                    self._print("# i-PI sent EXIT")
                    break
                elif hdr == b"":
                    break
                else:
                    raise RuntimeError(f"unexpected i-PI message {hdr!r}")
        except ConnectionError as err:
            self._print(f"# connection closed: {err}")
        finally:
            self.stats["t_total"] = time.perf_counter() - t_start
            self.sock.close()
        return self.stats


# ----------------------------------------------------------------------------- command line
def _engine_factory(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    from ..md.forcefield import MDSettings
    from ..md.simulation import _dedupe
    from ..param import read_prmtop_pgm
    from ..system import System
    from .engine import PGMEngine

    kw = json.loads(args.settings) if args.settings else {}
    if args.precision:
        kw["precision"] = args.precision
    settings = MDSettings(**kw)
    templates = None
    if args.template:
        from ..md.flexible import FlexibleTemplate

        tpl = FlexibleTemplate.load(args.template)
        n_mol = args.nmol
        if n_mol is None:
            raise SystemExit("--template needs --nmol (number of copies of the template molecule)")
        sys_ = System([tpl.pgm] * n_mol)
        templates = [tpl] * n_mol
    else:
        sys_ = System(_dedupe(read_prmtop_pgm(args.prmtop, first_residue_only=False)))

    def make(pos, cell):
        return PGMEngine(sys_, pos, cell, settings, templates=templates, slots=args.slots, stress=args.stress)

    return make


def main(argv=None):
    p = argparse.ArgumentParser(description="pGM (pgm_jax) client for i-PI")
    p.add_argument("--prmtop", help="pGM prmtop (rigid-molecule model; atoms in i-PI's order)")
    p.add_argument("--template", help="FlexibleTemplate file: --nmol copies of one flexible molecule")
    p.add_argument("--nmol", type=int)
    p.add_argument("--address", default="localhost")
    p.add_argument("--port", type=int, default=31415)
    p.add_argument("--unix", action="store_true")
    p.add_argument("--slots", type=int, default=1, help="induced-dipole histories (beads sent one by one)")
    p.add_argument(
        "--stress",
        default=None,
        choices=("atomic", "molecular"),
        help="virial: atomic (default with --template) or molecular (default for the rigid-molecule model)",
    )
    p.add_argument("--precision", choices=("mixed", "double"))
    p.add_argument("--settings", help='MDSettings as JSON, e.g. \'{"dipole_tol": 1e-5, "cutoff": 0.9}\'')
    p.add_argument("--no-virial", action="store_true")
    p.add_argument("--no-vmap", action="store_true", help="evaluate the structures of a batch one by one")
    p.add_argument("--no-dipole", action="store_true", help="no cell dipole in the extras")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if not (args.prmtop or args.template):
        p.error("--prmtop or --template is required")
    client = IPIClient(
        _engine_factory(args),
        args.address,
        args.port,
        args.unix,
        virial=not args.no_virial,
        verbose=args.verbose,
        vmap_beads=not args.no_vmap,
        dipole=not args.no_dipole,
    )
    st = client.run()
    eng = client.engine
    if eng is not None and hasattr(eng, "stats"):
        st = dict(st, engine=eng.stats)
    print("# " + json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
