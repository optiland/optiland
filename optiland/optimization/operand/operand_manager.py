"""Manages the collection of operands used during optimization."""

from __future__ import annotations

from optiland.optimization.operand import Operand


class OperandManager:
    """Manages a list of operands in a merit function.

    Attributes:
        operands (list): A list of Operand objects.

    Methods:
        add(operand_type, target, weight=1, input_data={}): Add an operand to
            the merit function.
        clear(): Clear all operands from the merit function.

    """

    def __init__(self):
        self.operands = []

    def add(
        self,
        operand_type=None,
        target=None,
        min_val=None,
        max_val=None,
        weight=1,
        input_data=None,
    ):
        """Add an operand to the merit function

        Args:
            operand_type (str): The type of the operand.
            target (float): The target value of the operand.
            min_val (float): The operand should stay above this
                value (inequality operand).
            max_val (float): The operand should stay below this
                value (inequality operand).
            weight (float): The weight of the operand.
            input_data (dict): Additional input data for the operand.

        """
        if input_data is None:
            input_data = {}
        self.operands.append(
            Operand(operand_type, target, min_val, max_val, weight, input_data),
        )

    def clear(self):
        """Clear all operands from the merit function"""
        self.operands = []

    def __iter__(self):
        """Return the iterator object itself"""
        self._index = 0
        return self

    def __next__(self):
        """Return the next variable in the list"""
        if self._index < len(self.operands):
            result = self.operands[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __len__(self):
        """Return the number of operands in the list"""
        return len(self.operands)

    def __getitem__(self, index):
        """Return the operand at the specified index"""
        return self.operands[index]

    def __setitem__(self, index, value):
        """Set the operand at the specified index"""
        if not isinstance(value, Operand):
            raise ValueError("Value must be an instance of Operand")
        self.operands[index] = value

    def __delitem__(self, index):
        """Delete the operand at the specified index"""
        del self.operands[index]
