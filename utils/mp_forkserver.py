# mp_forkserver.py
import multiprocessing as mp
import warnings
import traceback

def ensure_start_method(preferred: str = "forkserver",
                        fallback: str = "spawn",
                        force: bool = True) -> str:
    """
    안전하게 multiprocessing start method를 설정하고, 실제 사용 중인 방법을 반환.
    - preferred가 플랫폼에서 지원되지 않으면 fallback으로 자동 전환
    - 이미 설정돼 있으면 그대로 두고 경고만 출력
    - 예외 발생 시 합리적인 대안으로 폴백
    """
    chosen = None
    try:
        current = mp.get_start_method(allow_none=True)
        available = set(mp.get_all_start_methods())

        # 1) preferred 지원 여부 점검
        if preferred not in available:
            warnings.warn(
                f"Start method '{preferred}' not available on this platform; "
                f"falling back to '{fallback}'. available={sorted(available)}"
            )
            preferred = fallback if fallback in available else (
                "spawn" if "spawn" in available else next(iter(available))
            )

        # 2) 아직 미설정이면 선호 방식으로 설정
        if current is None:
            mp.set_start_method(preferred, force=force)
            chosen = preferred
        else:
            # 이미 설정됨 → 그대로 사용(충돌 방지)
            chosen = current
            if current != preferred:
                warnings.warn(
                    f"Start method already set to '{current}', leaving as-is "
                    f"(requested '{preferred}')."
                )

    except RuntimeError as e:
        # 보통 "context has already been set" 케이스
        chosen = mp.get_start_method()
        warnings.warn(f"Could not set start method ({e}). Using existing '{chosen}'.")

    except Exception as e:
        # 예기치 못한 오류 → 합리적 폴백 시도
        warnings.warn(
            "Unexpected error while setting start method: "
            f"{e}\n{traceback.format_exc()}"
        )
        try:
            if mp.get_start_method(allow_none=True) is None:
                if fallback in mp.get_all_start_methods():
                    mp.set_start_method(fallback, force=True)
                    chosen = fallback
                else:
                    # 최후의 수단: 사용 가능 목록 중 하나
                    any_method = next(iter(mp.get_all_start_methods()))
                    mp.set_start_method(any_method, force=True)
                    chosen = any_method
            else:
                chosen = mp.get_start_method()
        except Exception:
            # 정말 최후의 비상구
            chosen = "spawn"

    return chosen