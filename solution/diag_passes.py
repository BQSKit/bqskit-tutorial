import logging

from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import CircuitGate, CNOTGate

_logger = logging.getLogger('bqskit.diag_passes')

class PrintCNOTsPass(BasePass):
    async def run(self, circuit, data) -> None:
        _logger.info(f"BQSKit step, current CNOT count: {circuit.count(CNOTGate())}")
        
class PrintPartitionInfoPass(BasePass):
    async def run(self, circuit, data):
        acm = 0
        for op in circuit:            
            if not isinstance(op.gate, CircuitGate):
                _logger.info("Circuit not yet partitioned")
                return

            acm += op.gate._circuit.num_operations

        _logger.info(f"Number of partitions: {circuit.num_operations}")
        _logger.info(f"Average block size: {acm / circuit.num_operations}")