from typing import List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class VoltageViolation(BaseModel):
    bus_idx: int
    v_mag_pu: float


class ThermalViolation(BaseModel):
    name: str
    from_bus_idx: int
    to_bus_idx: int
    loading_percent: float


class Violation(BaseModel):
    voltage: List[VoltageViolation]
    thermal: List[ThermalViolation]
    disconnected_buses: List[int]


class IterationResult(BaseModel):
    iter: int
    executed_actions: List[Dict[str, Any]]
    viola_before: Violation
    viola_after: Violation


class Load(BaseModel):
    name: str
    bus_idx: int
    p_mw: float
    q_mvar: float
    curtailable: bool = Field(False)


class Generator(BaseModel):
    name: str
    bus_idx: int
    p_mw: float
    q_mvar: float
    controllable: bool = Field(False)


class Bus(BaseModel):
    idx: int
    name: str
    v_mag_pu: float


class Switch(BaseModel):
    from_bus_idx: int
    to_bus_idx: int
    name: str
    closed: bool
    controllable: bool = Field(True)


class Line(BaseModel):
    name: str
    from_bus_idx: int
    to_bus_idx: int
    loading_percent: float


class NetworkState(BaseModel):
    buses: List[Bus]
    switches: List[Switch]
    loads: List[Load]
    generators: List[Generator]
    lines: List[Line]


class ActionEffectiveness(BaseModel):
    action_name: str
    action_params: Dict[str, Any]
    network_id: str
    network_state: NetworkState
    viola_before: Violation
    viola_after: Violation


class State(BaseModel):
    # Core file paths
    network_file_path: Optional[str] = Field("./")  # Original network file
    org_network_copy_file_path: Optional[str] = Field("./")
    editing_network_file_path: Optional[str] = Field(
        "./"
    )  # Current working network file
    work_dir: Optional[str] = Field("./")  # Working directory for temporary files

    # Iteration tracking
    iteration_results: Optional[List[IterationResult]] = Field(default_factory=list)

    messages: Annotated[list, add_messages] = Field(default_factory=list)

    summary: Optional[str] = Field("")

    explanation: Optional[str] = Field("")

    @property
    def all_executed_actions(self) -> List[Dict[str, Any]]:
        acts = []
        for _iter in self.iteration_results:
            acts.extend(_iter.executed_actions)
        return acts

    @property
    def iter_num(self) -> int:
        return len(self.iteration_results) if self.iteration_results else 0
