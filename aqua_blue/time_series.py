from typing import IO, Union
from pathlib import Path

from dataclasses import dataclass
import numpy as np
from datetime import datetime
import pandas as pd

from numpy.typing import NDArray


@dataclass
class TimeSeries:

    dependent_variable: NDArray
    times: NDArray

    def __post_init__(self):
        
        if isinstance(self.times[0], datetime):
            self.times = np.array([t.timestamp() for t in self.times], dtype=np.float64)   
        timesteps = np.diff(self.times)

        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

    def save(self, file: IO, header="", delimiter=","):
        np.savetxt(
            file,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:

        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):

        data = pd.read_csv(fp)
        # Check if time column contains datetime values
        if isinstance(data.iloc[0, time_index], str):  
            data.iloc[:, time_index] = pd.to_datetime(data.iloc[:, time_index])
            times = data.iloc[:, time_index].apply(lambda x: x.timestamp()).values  # Convert to UNIX timestamps
        else:
            times = data.iloc[:, time_index].values  # Already numerical
        return cls(
            dependent_variable=data.delete(data.columns[time_index], axis=1).values, 
            times=data[:, time_index]
        )

    @property
    def timestep(self) -> float:

        return self.times[1] - self.times[0]
    
    def to_datetime(self):
        """
        Convert stored UNIX timestamps back to datetime objects.
        """
        return [datetime.utcfromtimestamp(t) for t in self.times]

    def __eq__(self, other) -> bool:

        return bool(np.all(self.times == other.times) and np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))
