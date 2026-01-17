# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(
        self,
        error_message: str | None = None,
        *args,
        suppress_context: bool = False,
        **kwargs,
    ):
        # Include the actual error message if provided
        if error_message:
            message = f"EngineCore error: {error_message}"
        else:
            message = (
                "EngineCore encountered an issue. "
                "See stack trace (above) for the root cause."
            )

        super().__init__(message, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
