# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Original code: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
#
# Modifications:
# - Modified optimizer to AdamW with added noise
# - Added temperature hyperparameter to control noise intensity

from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled
from dataclasses import dataclass, field
from transformers import TrainingArguments

from NoisyAdamW import NoisyAdamW

@dataclass
class CustomTrainingArguments(TrainingArguments):
    temperature: float = field(
        default=0.1,
        metadata={"help": "hyperparameter to control noise intensity"}
    )

class TrainerWithNoisyAdamW(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
        self.optimizer = NoisyAdamW(optimizer_grouped_parameters,
                                    self.args.learning_rate,
                                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                                    eps=self.args.adam_epsilon,
                                    weight_decay=self.args.weight_decay,
                                    temperature=self.args.temperature)
        
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
