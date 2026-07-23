# PostToolUse hook 반복 실패 원인 분석

- 작성일: 2026-07-23 (Asia/Seoul)
- 대상 저장소: `/home/pakchu/rllm`
- 대상 세션: `019ca270-ed25-73f3-af31-f5984f8742a1`
- 조사 범위: Codex/OMX hook 등록, 실행 경로, 설치·캐시 버전, 현재 세션 로그

## 결론

현재 반복되는 `PostToolUse hook (failed) / hook exited with code 1`의 직접 원인은 **실행 중인 Codex 세션이 이미 삭제된 OMX 플러그인 캐시 `0.20.2`의 hook 파일을 계속 호출하는 것**이다.

동시에 다음 두 hook 공급원이 모두 활성화되어 있다.

1. 프로젝트의 레거시 `.codex/hooks.json`
2. 사용자 범위의 OMX 플러그인 hook

따라서 도구 이벤트마다 hook이 2개 실행되고, 그중 플러그인 쪽이 사라진 `0.20.2` 경로를 호출하며 매번 실패한다. 프로젝트 hook은 현재 존재하는 글로벌 OMX `0.20.3` 스크립트를 호출하므로 통과한다. 이것이 **모든 PreToolUse/PostToolUse마다 실패 메시지가 반복되는 이유**다.

## Ranked synthesis

| 순위 | 설명 | 신뢰도 | 근거 |
|---:|---|---|---|
| 1 | 현재 세션의 플러그인 hook 경로가 삭제된 `0.20.2` 캐시를 가리킨다. | 높음 | 현재 pane/rollout의 `MODULE_NOT_FOUND`; 동일 경로 수동 재현 `rc=1`; 설치된 캐시는 `0.20.3`만 존재 |
| 2 | 프로젝트 hook과 플러그인 hook이 중복 등록되어 이벤트당 2회 실행된다. | 높음 | 두 별도 설정에 `PostToolUse`가 존재; 사용자 config에 두 trust entry가 존재; Codex 로그에 `hook/started`/`hook/completed`가 쌍으로 기록됨 |
| 3 | 1 MiB hook payload 제한이 반복 실패의 원인이다. | 낮음/기각 | 제한은 존재하지만 작은 read-only probe에서도 stale `0.20.2`만 실패하고 현재 두 실행 경로는 `rc=0`; 반복 실패가 출력 크기와 무관하게 발생 |

## 직접 증거

### 1. 현재 세션이 존재하지 않는 `0.20.2` 파일을 호출한다

현재 tmux pane과 rollout에는 다음 오류가 기록되어 있다.

```text
PostToolUse hook (failed)
error: hook exited with code 1

Error: Cannot find module '/home/pakchu/.codex/plugins/cache/oh-my-codex-local/oh-my-codex/0.20.2/hooks/codex-native-hook.mjs'
code: 'MODULE_NOT_FOUND'
```

근거 위치:

- Codex rollout: `/home/pakchu/.codex/sessions/2026/02/28/rollout-2026-02-28T13-10-35-019ca270-ed25-73f3-af31-f5984f8742a1.jsonl:262207`
- 같은 기록에 `PreCompact`와 `PostCompact`도 동일한 `0.20.2` 경로 때문에 실패한 내용이 포함되어 있다.

현재 디스크에 남은 플러그인 캐시 버전은 `0.20.3` 하나뿐이다.

```text
$ find ~/.codex/plugins/cache/oh-my-codex-local/oh-my-codex \
    -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
0.20.3
```

### 2. 세션 시작 후 plugin cache가 교체되었다

현재 Codex 프로세스 시작 시각:

```text
Wed Jul 22 22:47:15 2026
```

현재 `0.20.3` 캐시 설치 시각:

```text
2026-07-22 23:31:03 +0900
```

즉, 현재 프로세스가 시작된 뒤 plugin cache가 갱신되었다. 갱신 로그도 현재 세션이 이전 plugin registry를 메모리에 유지할 수 있으므로 새 세션을 시작하라고 명시한다.

- `.omx/logs/update-2026-07-22T14-19-36-494Z.log:42-49`
  - stale plugin discovery cache 1개를 무효화
  - `0.20.3` cache 설치
  - 현재 세션은 in-memory plugin registry를 유지할 수 있으므로 새 세션 필요

따라서 다음 순서가 성립한다.

1. Codex가 plugin `0.20.2` hook 절대경로를 메모리에 적재
2. OMX setup/update가 `0.20.2` cache를 제거하고 `0.20.3`을 설치
3. 실행 중인 Codex는 registry를 다시 읽지 않고 기존 `0.20.2` 경로를 계속 실행
4. 매 hook 이벤트마다 Node가 `MODULE_NOT_FOUND`로 종료 코드 1 반환

### 3. 현재 hook은 두 공급원에서 동시에 등록되어 있다

프로젝트 레거시 hook:

- `.codex/hooks.json:24-31`
- 실행 명령: `/usr/bin/node /home/pakchu/.npm-global/lib/node_modules/oh-my-codex/dist/scripts/codex-native-hook.js`

플러그인 hook:

- `/home/pakchu/.codex/plugins/cache/oh-my-codex-local/oh-my-codex/0.20.3/hooks/hooks.json:24-31`
- 실행 명령: `node "${PLUGIN_ROOT}/hooks/codex-native-hook.mjs"`

사용자 Codex 설정에도 양쪽 hook trust state가 모두 존재한다.

- 프로젝트 hook trust: `/home/pakchu/.codex/config.toml:30-49`
- 플러그인 hook trust: `/home/pakchu/.codex/config.toml:51-70`
- plugin hook 활성화: `/home/pakchu/.codex/config.toml:72-84`

Codex app-server 로그에서도 같은 시각에 `hook/started`와 `hook/completed`가 각각 2개씩 연속 기록된다. 예:

```text
312165810 hook/started
312165811 hook/started
312165812 hook/completed
312165813 hook/completed
```

로그 DB: `/home/pakchu/.codex/logs_2.sqlite`

현재 pane도 `Running 2 PostToolUse hooks`라고 표시한다. 따라서 중복 실행은 추정이 아니라 관측된 동작이다.

### 4. 동일 payload 재현 결과

작은 synthetic `PostToolUse` payload를 세 경로에 동일하게 전달했다.

| 실행 경로 | 종료 코드 | stdout | 결과 |
|---|---:|---|---|
| 삭제된 plugin `0.20.2` 경로 | 1 | 없음 | `MODULE_NOT_FOUND` 재현 |
| 현재 plugin `0.20.3` wrapper | 0 | `{}` | 정상 |
| 현재 global OMX `0.20.3` hook | 0 | `{}` | 정상 |

이 재현은 Node 자체, payload 형식, 현재 OMX `0.20.3` 구현이 주원인이 아님을 보여준다. 실패를 결정하는 차이는 **stale `0.20.2` 파일 경로**다.

## 왜 `.omx/logs/native-hook-*.jsonl`에 오류가 없는가

OMX native hook은 내부 dispatch 예외를 `.omx/logs/native-hook-YYYY-MM-DD.jsonl`에 기록한다.

- `/home/pakchu/.npm-global/lib/node_modules/oh-my-codex/dist/scripts/codex-native-hook.js:19433-19446`

하지만 이번 오류는 `codex-native-hook.mjs`가 로드되기도 전에 Node module loader에서 발생한다. 따라서 OMX 코드의 오류 logger에 도달하지 못하며, native-hook 오류 파일이 생성되지 않는 것이 현재 증상과 일치한다.

## 기각한 대안 원인

### Hook stdin 1 MiB 제한

현재 OMX에는 hook stdin JSON 최대 크기 `1,048,576` bytes 제한이 있다.

- `/home/pakchu/.npm-global/lib/node_modules/oh-my-codex/dist/scripts/hook-payload-guard.js:1-3`

큰 도구 출력에서는 별도 `native_hook_stdin_oversized`가 발생할 수 있다. 그러나 이것은 이번 반복 실패의 주원인이 아니다.

- 매우 작은 probe에서도 stale `0.20.2` 경로는 똑같이 `rc=1`이다.
- 현재 `0.20.3` plugin/global 경로는 같은 probe에서 모두 `rc=0`이다.
- 실제 pane의 구체적 오류는 oversized가 아니라 `MODULE_NOT_FOUND`다.

따라서 payload 제한은 별개의 잠재 리스크이며, 현재의 매-tool 반복 실패를 설명하지 않는다.

## 영향

- 도구 호출 자체는 대체로 성공하지만 각 PreToolUse/PostToolUse 뒤에 실패 메시지가 추가된다.
- 성공하는 레거시 프로젝트 hook이 남아 있어 OMX의 일부 side effect는 계속 수행될 수 있다.
- 플러그인 hook 경로는 매번 실패하므로 plugin 쪽 lifecycle 처리는 누락된다.
- 중복 실행 구조를 그대로 두면 restart 후에는 양쪽 hook이 모두 성공하면서 동일 side effect가 2회 실행될 가능성이 있다.
- 현재 문제는 trading 코드나 alpha 연구 결과의 오류가 아니라 Codex/OMX orchestration 계층의 설치·세션 수명 문제다.

## 권장 조치

### 즉시 조치

1. 현재 Codex/OMX 세션을 종료하고 새 세션을 시작한다.
2. 새 세션에서 plugin registry가 `0.20.3` 경로를 로드했는지 간단한 tool call로 확인한다.

현재 프로세스는 hook registry를 메모리에 보유하고 있으므로 파일만 갱신해도 기존 세션의 stale 절대경로는 바뀌지 않는다.

### 영구 조치

Plugin mode를 기준으로 hook 공급원을 하나로 정리한다.

- 유지: `/home/pakchu/.codex/config.toml`의 `plugin_hooks = true`와 `oh-my-codex@oh-my-codex-local` plugin
- 제거 또는 보관: 프로젝트의 레거시 `.codex/hooks.json`
- 정리 대상: 프로젝트 `.codex/config.toml`의 `hooks = true`와 프로젝트 hook trust entries
- 정리 대상: 사용자 `/home/pakchu/.codex/config.toml`에 남은 `.codex/hooks.json:*` trust entries

Trust entry 자체는 실행 명령이 아니지만, 레거시 hook이 계속 신뢰·활성화되는 상태를 보존하므로 함께 정리하는 편이 명확하다.

### 업그레이드 운영 규칙

- `omx setup` 또는 plugin cache 갱신 후에는 실행 중이던 Codex 세션을 재시작한다.
- 장시간 세션 중 plugin cache를 교체해야 한다면 이전 버전 cache를 즉시 삭제하지 않거나, 교체 직후 세션을 재시작한다.
- 재시작 후 `omx doctor`로 cache와 hook 공급원이 하나인지 확인한다.

## 현재 적용 상태

이 문서는 원인 분석 결과만 기록한다. 사용자 홈 설정과 프로젝트 hook 파일의 삭제·수정은 수행하지 않았다. 세션 재시작과 hook 단일화 후에는 아래 조건을 만족해야 해결된 것으로 본다.

1. 단일 tool call에서 `Running 1 PostToolUse hook` 또는 중복 없는 단일 lifecycle 실행
2. `PostToolUse hook (failed)` 미발생
3. `0.20.2/hooks/codex-native-hook.mjs` 참조 미발생
4. 현재 `0.20.3` hook이 종료 코드 0 반환

## 버전 스냅샷

```text
oh-my-codex: 0.20.3
Codex CLI:    0.144.6
Node.js:      22.22.0
plugin cache: 0.20.3 only
```
