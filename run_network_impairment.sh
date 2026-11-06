#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ====== User-editable defaults ======
DEVICE="${DEVICE:?Set DEVICE to the interface carrying Duckiebot traffic}"               # network interface to shape
SERVER_SCRIPT="${SERVER_SCRIPT:-gpu_server.py}"
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_adversary}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_adversary}"

# Default per-profile runtime (seconds) when running single profile
DEFAULT_RUNTIME_SINGLE=90

# When running the full set, pick these runtimes (secs)
RUNTIME_BASELINE=30
RUNTIME_SIMPLE=90
RUNTIME_MIXED=180

# ====== Helpers ======
tc_clear() {
    sudo tc qdisc del dev "$DEVICE" root 2>/dev/null || true
}

run_profile() {
    local profile_name="$1"
    local tc_cmd="$2"
    local runtime="$3"

    echo "[RUN] profile=$profile_name tc=\"$tc_cmd\" runtime=${runtime}s"
    tc_clear
    if [[ -n "$tc_cmd" && "$tc_cmd" != "none" ]]; then
        sudo bash -c "$tc_cmd"
    fi

    export NET_PROFILE="$profile_name"
    export TC_CMD="$tc_cmd"
    export EXPERIMENT_ID="$EXPERIMENT_ID"
    export RUN_ID="$(date +%Y%m%d_%H%M%S)"
    export TRIAL_ID="$profile_name"
    export OUTPUT_DIR="$OUTPUT_ROOT/$profile_name"
    mkdir -p "$OUTPUT_DIR"

    # if timeout available, use it to bound the run
    if command -v timeout >/dev/null 2>&1; then
        timeout "$runtime" python3 "$SERVER_SCRIPT" || true
    else
        echo "[WARN] timeout not found — running server without time limit (you'll need to Ctrl+C)"
        python3 "$SERVER_SCRIPT"
    fi

    # clean up
    tc_clear
    echo "[DONE] profile=$profile_name"
    echo
    sleep 3
}

trap tc_clear EXIT INT TERM

# ====== Build single profile from CLI args ======
usage() {
    cat <<EOF
Usage:
  ./run_network_impairment.sh                # run default suite
  ./run_network_impairment.sh delay 100      # single: add 100ms delay
  ./run_network_impairment.sh loss 5         # single: 5% packet loss
  ./run_network_impairment.sh jitter 150 50  # single: mean=150ms std=50ms
  ./run_network_impairment.sh bw 1mbit       # single: 1mbit bandwidth limit
  ./run_network_impairment.sh mix "delay=200 loss=5"  # single compound profile
  Optional add runtime (seconds) as last arg:
    ./run_network_impairment.sh delay 100 120
EOF
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# If no args: run the full default set
if [[ $# -eq 0 ]]; then
    declare -A PROFILES
    PROFILES["baseline"]="tc qdisc del dev $DEVICE root 2>/dev/null || true"
    PROFILES["delay100"]="tc qdisc add dev $DEVICE root netem delay 100ms"
    PROFILES["jitter150_50"]="tc qdisc add dev $DEVICE root netem delay 150ms 50ms distribution normal"
    PROFILES["bw_1mbit"]="tc qdisc add dev $DEVICE root netem rate 1mbit"
    PROFILES["bw_900kbit"]="tc qdisc add dev $DEVICE root netem rate 900kbit"
    PROFILES["bw_750kbit"]="tc qdisc add dev $DEVICE root netem rate 750kbit"
    PROFILES["bw_600kbit"]="tc qdisc add dev $DEVICE root netem rate 600kbit"
    PROFILES["bw_500kbit"]="tc qdisc add dev $DEVICE root netem rate 500kbit"
    PROFILES["delay200_loss5"]="tc qdisc add dev $DEVICE root netem delay 200ms loss 5%"
    PROFILES["delay150_jitter30_dup2"]="tc qdisc add dev $DEVICE root netem delay 150ms 30ms duplicate 2%"
    PROFILES["delay100_loss5_reorder25"]="tc qdisc add dev $DEVICE root netem delay 100ms 40ms reorder 25% 50%"

    for PROFILE in "${!PROFILES[@]}"; do
        # choose runtime depending on complexity
        if [[ "$PROFILE" == "baseline" ]]; then
            RUNTIME="$RUNTIME_BASELINE"
        elif [[ "$PROFILE" =~ delay|loss|jitter|bw ]]; then
            RUNTIME="$RUNTIME_SIMPLE"
        else
            RUNTIME="$RUNTIME_MIXED"
        fi
        run_profile "$PROFILE" "${PROFILES[$PROFILE]}" "$RUNTIME"
    done

    exit 0
fi

# If args provided: build single profile
MODE="${1:-}"
shift

case "$MODE" in
    delay)
        if [[ $# -lt 1 ]]; then usage; fi
        VALUE="$1"; shift
        PROFILE_NAME="delay${VALUE}"
        TC_CMD="tc qdisc add dev $DEVICE root netem delay ${VALUE}ms"
        ;;

    loss)
        if [[ $# -lt 1 ]]; then usage; fi
        VALUE="$1"; shift
        PROFILE_NAME="loss${VALUE}"
        # allow percent or plain number
        TC_CMD="tc qdisc add dev $DEVICE root netem loss ${VALUE}%"
        ;;

    jitter)
        if [[ $# -lt 2 ]]; then usage; fi
        MEAN="$1"; STD="$2"; shift 2
        PROFILE_NAME="jitter_${MEAN}_${STD}"
        TC_CMD="tc qdisc add dev $DEVICE root netem delay ${MEAN}ms ${STD}ms distribution normal"
        ;;

    bw)
        if [[ $# -lt 1 ]]; then usage; fi
        RATE="$1"; shift
        PROFILE_NAME="bw_${RATE}"
        TC_CMD="tc qdisc add dev $DEVICE root netem rate ${RATE}"
        ;;

    mix)
        # pass a quoted string: e.g. "delay=200 loss=5 reorder=25,50"
        if [[ $# -lt 1 ]]; then usage; fi
        ARGSTR="$1"; shift
        # parse space-separated key=val pairs
        PROFILE_NAME="mix"
        TC_CMD="tc qdisc add dev $DEVICE root netem"
        for kv in $ARGSTR; do
            k="${kv%=*}"
            v="${kv#*=}"
            case "$k" in
                delay) TC_CMD+=" delay ${v}ms"; PROFILE_NAME+="_d${v}";;
                loss) TC_CMD+=" loss ${v}%"; PROFILE_NAME+="_l${v}";;
                reorder) TC_CMD+=" reorder ${v}"; PROFILE_NAME+="_r${v}";; # user-supplied format ok
                duplicate) TC_CMD+=" duplicate ${v}%"; PROFILE_NAME+="_dup${v}";;
                *) echo "[WARN] unknown mix key: $k";;
            esac
        done
        ;;

    *)
        echo "[ERROR] unknown mode: $MODE"
        usage
        ;;
esac

# optional runtime override as last argument
RUNTIME="$DEFAULT_RUNTIME_SINGLE"
if [[ $# -ge 1 ]]; then
    RUNTIME="$1"
fi

# Run the single generated profile
run_profile "$PROFILE_NAME" "$TC_CMD" "$RUNTIME"

