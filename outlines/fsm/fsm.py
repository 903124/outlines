from typing import TYPE_CHECKING, List, NewType, Optional, Protocol

import interegular

from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer

FSMState = NewType("FSMState", int)


class FSM(Protocol):
    def next_instruction(self, state: FSMState) -> List[int]:
        ...

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        ...

    def is_final_state(self, state: FSMState) -> bool:
        ...


class StopAtTokenFSM(FSM):
    """FSM to generate text until a specified token id is generated or
    a specified number of tokens has been generated.

    Text is usually produced until the EOS token is generated by the
    model.

    """

    def __init__(
        self,
        tokenizer: "Tokenizer",
        stop_token_id: int,
        max_tokens: Optional[int] = None,
    ):
        self.stop_token_id = stop_token_id
        self.max_tokens = max_tokens
        self.num_tokens_generated = 0
        self.vocabulary = tokenizer.vocabulary.values()

    def next_instruction(self, state: FSMState) -> List[int]:
        """Generate a list of forbidden tokens for the next step.

        When in the initial state we allow every token to be generated.
        In the final state the only allowed token is `stop_token_id`.

        Parameters
        ----------
        state
            The current state of the FSM

        Returns
        -------
        A list that contains the tokens to mask.

        """
        if state == 0:
            return []
        else:
            return [
                token_id
                for token_id in self.vocabulary
                if token_id != self.stop_token_id
            ]

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

        The FSM stays in the initial state `0` unless the specified stop token
        has been generated or the maximum number of tokens has been reached. In
        which case the FSM moves to the final state `1`.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.

        """
        self.num_tokens_generated += 1

        if self.max_tokens is not None:
            if self.num_tokens_generated >= self.max_tokens:
                return FSMState(1)

        if token_id == self.stop_token_id:
            return FSMState(1)

        return FSMState(0)

    def is_final_state(self, state: FSMState) -> bool:
        """Determine whether the current state of the FSM is a final state."""

        if state == 1:
            return True
        else:
            return False


class RegexFSM(FSM):
    """FSM to generate text that is in the language of a regular expression."""

    def __init__(
        self,
        regex_string: str,
        tokenizer: "Tokenizer",
        max_tokens: Optional[int] = None,
    ):
        regex_pattern = interegular.parse_pattern(regex_string)
        regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
        (
            self.states_to_token_maps,
            self.empty_token_ids,
        ) = create_fsm_index_tokenizer(regex_fsm, tokenizer)

        if not any(
            regex_fsm.finals.intersection(v.values())
            for v in self.states_to_token_maps.values()
        ):
            raise ValueError(
                "The vocabulary does not allow us to build a sequence that matches the input regex"
            )

        self.final_states = regex_fsm.finals | {
            -1
        }  # Include the EOS token in final states
        self.max_tokens = max_tokens
        self.num_tokens_generated = 0
        self.vocabulary = tokenizer.vocabulary.values()
        self.end_token = tokenizer.eos_token_id

    def next_instruction(self, state: FSMState) -> List[int]:
        """Generate a list of forbidden tokens for the next step.

        The initialization of the FSM builds an index which maps FSM states to a
        map from authorized tokens to the state in which the FSM needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the FSM. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the FSM

        Returns
        -------
        A list that contains the tokens to mask.

        """
        next_tokens_to_end_states = self.states_to_token_maps.get(state)

        if next_tokens_to_end_states is None:
            authorized_tokens = [self.end_token]
        else:
            authorized_tokens = list(next_tokens_to_end_states.keys())

        forbidden_tokens = [
            token for token in self.vocabulary if token not in authorized_tokens
        ]

        return list(forbidden_tokens)

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

        We use the index to determine to which state the FSM should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.

        """
        self.num_tokens_generated += 1

        if self.max_tokens is not None:
            if self.num_tokens_generated == self.max_tokens:
                return FSMState(-1)

        if token_id == self.end_token:
            return FSMState(-1)

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = -1

        return FSMState(next_state)

    def is_final_state(self, state: FSMState) -> bool:
        """Determine whether the current state of the FSM is a final state."""

        if state in self.final_states:
            return True

        return False
