


def stage_id2resolution(stage_id):
    parts = stage_id.split('_')
    stage_word = None
    if "enc" in parts:
        stage_word = "enc"
    elif "dec" in parts:
        stage_word = "dec"
    elif "bottleneck" in parts:
        stage_word = "bottleneck"
    else:
        raise ValueError(f"Unknown stage_id: {stage_id}")
    stage_num = parts[parts.index(stage_word) + 1]
    if stage_word == "enc":
        stage_num = int(stage_num)
        if stage_num == 0:
            return 0
        elif stage_num == 1:
            return 1
    elif stage_word == "bottleneck":
        return 2
    elif stage_word == "dec":
        stage_num = int(stage_num)
        if stage_num == 0:
            return 1
        elif stage_num == 1:
            return 0
    raise ValueError(f"Unknown stage_id: {stage_id}")

if __name__ == "__main__":
    tests = [
        "0_enc_0_block_0_ss",
        "0_enc_1_block_0_ss",
        "0_bottleneck_block_0_ss",
        "0_dec_0_block_0_ss",
        "0_dec_1_block_0_ss",
    ]
    for test in tests:
        res = stage_id2resolution(test)
        print(f"{test} -> {res}")
