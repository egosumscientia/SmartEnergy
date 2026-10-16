"""Bucle opcional para reentrenar y recalcular metricas explicitamente."""
import argparse
import subprocess
import sys
import time


def run_cmd(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] {' '.join(cmd)} -> {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Loop de reentrenamiento y recalculo de metricas.")
    parser.add_argument("--interval", type=int, default=300, help="Segundos entre ciclos (default: 300).")
    args = parser.parse_args()

    print(f"[auto_train_loop] Iniciando loop cada {args.interval} segundos. Ctrl+C para salir.")
    try:
        while True:
            ok_train = run_cmd([sys.executable, "-m", "smartenergy.operations.train_model"])
            ok_detect = False
            if ok_train:
                ok_detect = run_cmd([sys.executable, "-m", "smartenergy.operations.detect_anomalies"])
            if ok_train and ok_detect:
                print("[auto_train_loop] OK: modelo entrenado y metricas recalculadas.")
            time.sleep(max(args.interval, 1))
    except KeyboardInterrupt:
        print("\n[auto_train_loop] Detenido por el usuario.")


if __name__ == "__main__":
    main()
