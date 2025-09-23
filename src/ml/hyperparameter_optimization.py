"""
Advanced Hyperparameter Optimization for CNN+LSTM Models - Task 5.5

This module implements comprehensive hyperparameter optimization using Optuna with:
- Multi-objective optimization balancing accuracy, training time, and model size
- Early pruning for efficiency across 1000+ trials
- Automated search for learning rates, architectures, and regularization
- Best configuration saving and final model retraining

Requirements: 3.4, 9.1
"""

import os
import sys
import json
import time
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig, create_hybrid_config
from src.ml.train_integrated_cnn_lstm import IntegratedCNNLSTMTrainer
from data.pipeline import create_data_loaders


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hyperparameter_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    # Study configuration
    study_name: str = "cnn_lstm_optimization"
    n_trials: int = 1000
    timeout: int = 172800  # 48 hours in seconds
    n_jobs: int = 1  # Parallel jobs
    
    # Pruning configuration
    pruning_strategy: str = "hyperband"  # "median", "hyperband", "none"
    min_resource: int = 10  # Minimum epochs before pruning
    max_resource: int = 100  # Maximum epochs for full training
    reduction_factor: int = 3  # Hyperband reduction factor
    
    # Multi-objective weights
    accuracy_weight: float = 0.5
    training_time_weight: float = 0.3
    model_size_weight: float = 0.2
    
    # Early stopping
    early_stopping_patience: int = 15
    min_epochs: int = 20
    max_epochs: int = 200
    
    # Data configuration
    validation_split: float = 0.2
    batch_size_range: Tuple[int, int] = (16, 128)
    
    # Search space bounds
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    dropout_rate_range: Tuple[float, float] = (0.1, 0.7)
    hidden_dim_range: Tuple[int, int] = (64, 512)
    
    # Output configuration
    results_dir: str = "hyperopt_results_task_5_5"
    save_top_k: int = 10
    plot_results: bool = True


class MultiObjectiveOptimizer:
    """
    Advanced multi-objective hyperparameter optimizer for CNN+LSTM models.
    
    Implements task 5.5 requirements:
    - Optuna-based optimization with 1000+ trials
    - Early pruning for efficiency
    - Multi-objective optimization (accuracy, time, size)
    - Best configuration saving and model retraining
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
        device: Optional[str] = None
    ):
        """Initialize the multi-objective optimizer"""
        self.config = config
        self.train_loader, self.val_loader, self.test_loader = data_loaders
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create results directory
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.best_trials = []
        self.optimization_history = []
        
        # Performance tracking
        self.trial_results = []
        self.start_time = None
        
        logger.info(f"Multi-objective optimizer initialized with {self.config.n_trials} trials")
        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner"""
        
        # Choose sampler based on problem complexity
        if self.config.n_trials > 500:
            sampler = TPESampler(
                n_startup_trials=50,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            )
        else:
            sampler = CmaEsSampler(
                n_startup_trials=20,
                restart_strategy="ipop"
            )
        
        # Choose pruner based on strategy
        if self.config.pruning_strategy == "hyperband":
            pruner = HyperbandPruner(
                min_resource=self.config.min_resource,
                max_resource=self.config.max_resource,
                reduction_factor=self.config.reduction_factor
            )
        elif self.config.pruning_strategy == "median":
            pruner = MedianPruner(
                n_startup_trials=20,
                n_warmup_steps=10,
                interval_steps=5
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction="maximize",  # Maximize composite score
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Created study with {type(sampler).__name__} sampler and {type(pruner).__name__} pruner")
        
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> HybridModelConfig:
        """Suggest hyperparameters for a trial"""
        
        # CNN hyperparameters
        cnn_num_filters = trial.suggest_categorical("cnn_num_filters", [32, 64, 128, 256])
        cnn_filter_sizes = trial.suggest_categorical(
            "cnn_filter_sizes", 
            [[3, 5], [3, 5, 7], [3, 5, 7, 11], [5, 7, 11, 15]]
        )
        cnn_attention_heads = trial.suggest_categorical("cnn_attention_heads", [4, 8, 16])
        
        # LSTM hyperparameters
        lstm_hidden_dim = trial.suggest_int(
            "lstm_hidden_dim", 
            self.config.hidden_dim_range[0], 
            self.config.hidden_dim_range[1],
            step=32
        )
        lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 4)
        lstm_bidirectional = trial.suggest_categorical("lstm_bidirectional", [True, False])
        lstm_use_attention = trial.suggest_categorical("lstm_use_attention", [True, False])
        lstm_use_skip_connections = trial.suggest_categorical("lstm_use_skip_connections", [True, False])
        
        # Fusion hyperparameters
        feature_fusion_dim = trial.suggest_int("feature_fusion_dim", 128, 512, step=64)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float(
            "learning_rate",
            self.config.learning_rate_range[0],
            self.config.learning_rate_range[1],
            log=True
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            self.config.dropout_rate_range[0],
            self.config.dropout_rate_range[1]
        )
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        # Optimizer hyperparameters
        optimizer_type = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Scheduler hyperparameters
        scheduler_type = trial.suggest_categorical(
            "scheduler", 
            ["cosine", "step", "exponential", "plateau"]
        )
        
        # Multi-task weights
        classification_weight = trial.suggest_float("classification_weight", 0.2, 0.8)
        regression_weight = 1.0 - classification_weight
        
        # Ensemble configuration
        num_ensemble_models = trial.suggest_int("num_ensemble_models", 3, 7)
        ensemble_dropout_rate = trial.suggest_float("ensemble_dropout_rate", 0.05, 0.3)
        
        # Create configuration
        config = HybridModelConfig(
            # Model architecture
            input_dim=11,  # Will be updated based on data
            cnn_filter_sizes=cnn_filter_sizes,
            cnn_num_filters=cnn_num_filters,
            cnn_use_attention=True,
            cnn_attention_heads=cnn_attention_heads,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_bidirectional=lstm_bidirectional,
            lstm_use_attention=lstm_use_attention,
            lstm_use_skip_connections=lstm_use_skip_connections,
            feature_fusion_dim=feature_fusion_dim,
            
            # Training configuration
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            classification_weight=classification_weight,
            regression_weight=regression_weight,
            
            # Ensemble configuration
            num_ensemble_models=num_ensemble_models,
            ensemble_dropout_rate=ensemble_dropout_rate,
            
            # Multi-task configuration
            num_classes=4,  # 4 regime classes
            regression_targets=2,  # Price and volatility
            
            # Other parameters
            sequence_length=60,
            prediction_horizon=10,
            use_monte_carlo_dropout=True,
            mc_dropout_samples=50,
            device=str(self.device)
        )
        
        # Store additional trial parameters
        trial.set_user_attr("optimizer_type", optimizer_type)
        trial.set_user_attr("weight_decay", weight_decay)
        trial.set_user_attr("scheduler_type", scheduler_type)
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Returns composite score balancing accuracy, training time, and model size.
        """
        trial_start_time = time.time()
        
        try:
            # Suggest hyperparameters
            config = self.suggest_hyperparameters(trial)
            
            # Create trainer
            trainer = IntegratedCNNLSTMTrainer(
                config=config,
                save_dir=str(self.results_dir / f"trial_{trial.number}"),
                device=str(self.device)
            )
            
            # Determine input dimension from data
            sample_batch = next(iter(self.train_loader))
            input_dim = sample_batch[0].shape[-1]  # Last dimension is features
            config.input_dim = input_dim
            
            # Build models
            trainer.build_models(input_dim)
            
            # Calculate model size (number of parameters)
            model_size = sum(p.numel() for p in trainer.hybrid_model.parameters())
            model_size_mb = model_size * 4 / (1024 * 1024)  # Approximate MB
            
            # Training with early stopping and pruning
            best_val_accuracy = 0.0
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Setup optimizer based on trial suggestion
            optimizer_type = trial.user_attrs.get("optimizer_type", "adamw")
            weight_decay = trial.user_attrs.get("weight_decay", 1e-5)
            
            if optimizer_type == "adam":
                optimizer = optim.Adam(
                    trainer.hybrid_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_type == "adamw":
                optimizer = optim.AdamW(
                    trainer.hybrid_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=weight_decay
                )
            else:  # sgd
                optimizer = optim.SGD(
                    trainer.hybrid_model.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
            
            # Setup scheduler
            scheduler_type = trial.user_attrs.get("scheduler_type", "cosine")
            if scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2
                )
            elif scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
            elif scheduler_type == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            else:  # plateau
                scheduler = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")
            
            # Training loop with pruning
            for epoch in range(self.config.max_epochs):
                # Training phase
                traidir']}")s_lts['result {resuctory:ts dire"Resul   print(f)
 alue:.4f}"trial'].v['best_sults {ree:or"Best sc    print(feted!")
complization nt("Optimri    
    p"
    )
estsk_5_5_t_results_tar="hyperopt_diults res       g
 testined for=2,  # Reducmeout_hours     ti
   or testingd f Reduce=100,  #   n_trials
     ",01-01te="2024-    end_da
    1",0-01-0ate="202     start_d  NVDA"],
 TSLA", ""MSFT", ", "OGLAAPL", "GO["   symbols=
     imization(rameter_optyperpas = run_hresult
     usage  # Exampleain__":
  = "__mf __name__ =

immary
  return su    
  ary)
ummite(s.wr   f f:
     'w') asmary.txt", ation_sum"optimizults_dir / ith open(res    wr'])
_dits['resultsh(resulPats_dir = esult   rary
 Save summ    # "
    
space
""ch fined seartion with reimizaional opting additr runnsideCon
4. ploymentuction der prodls foned moderetraiUse 
3. rectoryots/ diin the pllizations sua. Examine vijson
2ions.t_configuratns in besiguratiot confew the besRevi Steps
1. ext

## N*/model_retrained_: ed modelsain/
- Retr plotstions: Visualiza
-.csvsults: trial_rel resultsjson
- Trian_analysis.zatio: optimiesultsAnalysis rn
- jsoations.figurest_con burations:onfigBest c.pkl
- na_studyect: optutudy objd
- Seratees Gen# Fil
# += f"""mary   sum  
 
  ue}\n" {valparam}: {"-mmary += fsu
        s():.itemparamsst_trial. be, value infor param    
    """
iguration
nfst Co

## Be1f} hours:.] / 3600e'_timimizationopt {results['ation Time:ptimiznumber}
- Otrial.st_umber: {best Trial N}
- Beal.value:.4ftrie: {best_ Scormpositeest Cos
- B# Result}

#ize_weightl_smode={config.}, Sizeime_weightining_ttrame={config.ight}, Ti.accuracy_wecy={configAccuraights: ective We Multi-objtegy}
-straig.pruning_onfegy: {cng Stratruniurs
- P} ho/ 3600:.1ffig.timeout conout: {ls}
- Timen_tria: {config.rials- Total Ton
onfiguratiry

## Czation Summaeter OptimirparamLSTM Hype""
# CNN+y = f"mmar  
    su  rial']
t_ts['besesultial = rtr  best_
      ""
ation" the optimiz ofary reporterate a summ """Genstr:
   ) -> nfigimizationCoonfig: Opty], cDict[str, An(results: mary_reportnerate_sumgef _
de
rn results

    retu
    :.4f}")al'].valuets['best_triore: {resulsccomposite "Best o(fogger.inf
    ls_dir']}")sults['re: {resultlts saved toResu.info(f"
    logger")lly!ssfuucce sn completedtimizatioter oprameo("Hyperpa  logger.inf   
  nfig)
 s, coport(result_summary_reerateenmmary = _g
    sury reportate summa# Gener   
    imize()
 izer.opttimts = opul   restion
 miza# Run opti
    
     )r)
   st_loadeer, te val_loadr,den_loa=(trairsloadeata_      dig,
  onfconfig=c       timizer(
 tiveOpbjec= MultiOptimizer zer
    oimi# Create opt   
    s")
 batcher)} test (test_loade{lenr)} val, _loadevaltrain, {len(} ader)en(train_lored: {lrepa"Data pr.info(f logge  
   
    )
  32tch_size=    ba   h=60,
 e_lengtquencse    
    d_date,ate=end_d     endate,
   art_te=st  start_da      mbols,
=sybols       sym(
 are_datar.prepraine t =loader test_er,al_loadoader, vain_l
    tr    p"
    )
redata_pemp_ve_dir="t
        sationraepa for data prconfigry ,  # Temporad_config()rie_hybconfig=creat       iner(
 raMTedCNNLSTtegratrainer = In    ta...")
training dat"Preparing ger.info(    logata
epare d
    # Pr
    dir)ts_sul, rehourst_als, timeou(n_triconfigtimization_= create_opnfig on
    co configurationtioptimiza # Create    
   h")
 hours}t_ {timeoueout:ls}, Timiatrals: {n_"Tri.info(fogger
    lnd_date}")to {e_date} rtstage: { rannfo(f"Date logger.i
   ls}")bo: {symolsf"Symbr.info(
    loggen...")ioizatr optimerparameteTM hyp+LSing CNN"Startfo(   logger.in
    
 "
    ""trainingnd model resaving aguration onfi - Best czation
   optimibjective  - Multi-oy
   ciencr effipruning fo - Early    0+ trials
n with 100 optimizatioptuna-based
    - Os: requirement 5.5s all taskmention impleThis funct    
    ine.
 pipelmizationmeter optirpara hypeompleteRun c
    ""
    "]: Any Dict[str,"
) ->ts_task_5_5_resulperopt: str = "hy results_dir,
    48rs: int =out_houtime000,
    int = 1: ials
    n_tr-01", "2024-01: str =te
    end_da",2020-01-01tr = "t_date: s  starNVDA"],
  , """, "TSLA"MSFTGOOGL", ", "] = ["AAPL: List[strbolsymon(
    stimizatieter_ophyperparamdef run_    )


=True
sultst_re    plo=10,
    _ksave_top      s_dir,
  ultesr=rsults_di
        re),512ge=(64, dim_randden_ hi    
   0.7),, range=(0.1pout_rate_ro       d1e-2),
 -5, =(1egeing_rate_ran       learn, 128),
 nge=(16ze_rabatch_si        lit=0.2,
spidation_       val0,
 s=20  max_epoch=20,
      epochs     min_ce=15,
   ienpatping_early_stop
        weight=0.2,_size_      modelght=0.3,
  ime_weiining_t
        traght=0.5,_weiacy     accur  ,
 ion_factor=3  reduct      ce=100,
ax_resour    m=10,
    ource_res    min",
    yperbandgy="hing_strateprun     s=1,
   n_job       * 3600,
 urs eout_hotimeout=tim,
        =n_trials   n_trials
     %S')}",d_%H%Mime('%Y%m%strft().time.nowtion_{datem_optimizan_lst"cnudy_name=f    st
    (Configationrn Optimiz
    retu""
    ults" defalensibwith sefiguration mization conoptie """Creat  
  nConfig:izatio> Optimk_5_5"
) -_tasults_reshyperoptr: str = "ults_dires8,
    t = 4rs: inhouimeout_,
    t int = 1000s:
    n_trialonfig(ion_c_optimizatate cre
def    }

al
    t_totes': tesmpll_sa 'tota         
  se,g_reg_mn_mse': av 'regressio     ,
      acyracy': accur  'accu    
      rn {     retu          
 t_loader)
en(self.tesg_mse / lmse = revg_reg_l
        atato / test_ct_corre00.0 * testuracy = 1
        acc
        ()).itemgetsg_tarreeg_outputs, MSELoss()(rg_mse += nn.re       
         tricsegression me     # R           
             )
   um().item(s_targets).sed == clasredictrect += (pst_cor         te
       gets.size(0)_tarass += cltalst_to       te
         its.data, 1)ass_logh.max(cld = torc, predicte      _        metrics
  ion ificat Class #                  
        ata)
      _ = model(duts,_outps, reg_logit   class            
                 evice)
to(self.d_targets.argets = regg_t         re       lf.device)
(se_targets.to= classts class_targe               f.device)
 ela.to(sata = dat  d           
   oader:_lelf.testts in sargeeg_tgets, rtara, class_     for dat():
       radh.no_grcith to
        w     = 0.0
    reg_mse  = 0
      total  test_
       0rrect = t_co       tes
  = 0.0st_loss te       el.eval()
     mod  
       et"""
  est smodel on tte alua"""Ev]:
        tr, float-> Dict[sle)  nn.Modudel:elf, mo(sst_set_tevaluate_on_e   def 
    
         )evice)
elf.dce=str(s  devi        es=50,
  ropout_sampl  mc_d        
  e,Truut=lo_dropoe_monte_car      us,
      horizon=10diction_   pre        gth=60,
 en_lequence           ss
 rameter pa  # Other            
        ets=2,
  arggression_t  re        ses=4,
  num_clas            ion
ratconfiguk lti-tas   # Mu                
     ],
rate'ropout_ensemble_d=params['ropout_rate_d  ensemble       ],
   s'e_modelm_ensemblms['nuls=paramble_modem_ense       nuion
      configuratmble      # Ense       
    ,
       ight']cation_welassifi- params['cght=1.0 gression_wei     re       ],
t'_weighficationlassi'carams[on_weight=picatisif   clas,
         batch_size']params['batch_size=           te'],
 pout_ra['dro=paramste  dropout_ra          
ate'],learning_rams['pararning_rate=  le       
   ationing configur Train         #
             n_dim'],
  ature_fusiofedim=params['usion_ature_f       fe    ctions'],
 conneuse_skip_ams['lstm_ections=parnnip_couse_sk lstm_          ion'],
 tentuse_ats['lstm_ramention=paatte_lstm_us         nal'],
   rectios['lstm_bidiparamonal=_bidirecti  lstm         yers'],
 lstm_num_las['rampayers=la_num_    lstm       
 m'],idden_di_harams['lstm_dim=pddenlstm_hi        
    s'],ion_headttentms['cnn_ads=paration_heaten    cnn_at
        rue,ntion=Tten_use_at cn           ,
um_filters']ams['cnn_nrs=par_filten_num  cn         _sizes'],
 filternn_es=params['cfilter_siz        cnn_pdated
    # Will be uut_dim=11,        inp      cture
iteel arch       # Mod    ig(
 Conf HybridModel    return 
    s
       er_attral.usttrs = tri user_a    s
   l.param = tria params             
 "
 onfig""odelCbridMto Hyarameters rial pert tnv""Co       "onfig:
 ybridModelC.Trial) -> Hrial: optuna tnfig(self,l_to_co  def _tria
     d_results
 ineurn retraet    r    
    ")
    uccessfullydels smoults)} d_reslen(retraine {rained"Retfo(fogger.in        l      
t=2)
  , f, indend_resultsrainen.dump(ret     jso f:
        'w') as",lts.jsonodels_resued_mtrainrets_dir / "sulelf.reth open(s     wiults
   trained resve re   # Sa
     ue
        ontin   c        )
      {e}"i+1}:l {etrain modeFailed to r.error(f"erogg          l
      e:tion as t Excep  excep   
                      ")
 2f}%racy', 0):.t('accumetrics.geacy: {test_ccur Aest     f"T                     lly. "
 fucessned suc{i+1} retrai"Model fo(f   logger.in            
         
        ed_result)etrainpend(results.apd_rtraine       re       
              }
                _dir)
    tr(retrainel_path': s       'mod       s,
      t_metric testrics':t_me      'tes    
          results,ng_ training_results':aini     'tr               ,
g)dict(confi'config': as                
    alue,al.vricore': tinal_sig'or                   
 r,umbeal.n triinal_trial':rig      'o            
   + 1, 'rank': i            
       = {result ained_tr          re        
              _model)
id.hybrainer_set(tre_on_test_evaluatelf. smetrics =  test_            set
   on test luate   # Eva                   
       
      )           ce=50
  atien_stopping_ply  ear                00,
  hs=2_epoc   num              loader,
   f.val_         sel          er,
 in_load   self.tra                d_model(
 tegrateer.train_inults = traing_res    trainin       g)
     stoppinearly ning (no l trai     # Ful            
         im)
      ls(input_dld_moderainer.bui           t   
   modelld and train      # Bui        
                 input_dim
 t_dim = nfig.inpu         co     ape[-1]
  batch[0].shmple_put_dim = sa      in          der))
n_loaself.traiext(iter(ch = ne_batampl      s      sion
    ennput dim# Get i            
           
                    )ce)
     (self.devistr  device=         
         train_dir),retr(ve_dir=ssa                 nfig,
    config=co               er(
    ainSTMTregratedCNNL Int   trainer =          "
   _model_{i+1}"retrained/ fts_dir resulelf._dir = setrain  r          ner
    rai# Create t      
                        l)
  riaonfig(trial_to_c._tonfig = self           con
     nfiguraticoate   # Recre          
    :        try    
            er})...")
rial.numb(Trial {t}/3 g model {i+1inino(f"Retra  logger.inf          trials):
rate(top_ in enumerialr i, t     fo
           ]
= [ned_results   retrai   
      [:3]
     lsiaompleted_trls = c  top_tria
      rse=True)lue, revea t: t.vat(key=lambd.soralsed_triplet      comLETE]
  alState.COMP.Tri.trial == optuna if t.stateudy.trials.str t in selfls = [t fod_triaplete     comions
   at 3 configur  # Get top     
 
        ning...")h full traidels witng best moraininfo("Retlogger.i 
            urn []
         ret
      tudy.trials:self.sif not          
     ng"""
  raini with full tdelsmothe best "Retrain ""       ]]:
 nyict[str, AList[Dself) -> odels(est_mn_bretrai
    def _()
    lt.close  pht')
      s='tighencx_i bbo00,i=3ng", dpparison.p0_comr / "top_1(plots_di plt.savefig)
       ht_layout(     plt.tig      
   (MB)')
  'Model Size t_ylabel(, 1].sees[1   axank')
     al Rabel('Tri_xl 1].set     axes[1,zes')
   odel Si10 MTop set_title('axes[1, 1].       e_mb'])
 0['model_sizop_1top_10)), t(len(range, 1].bar(  axes[1      es
l siz      # Mode      
  )')
  e (simaining TTrt_ylabel('s[1, 0].se  axek')
      Trial Ranset_xlabel('axes[1, 0].')
        ng Times10 Trainitle('Top  0].set_ti[1,  axes   e'])
   imtraining_t['), top_100)op_1range(len(tbar(0].axes[1,       times
   # Training 
       
        uracy (%)')Accet_ylabel('xes[0, 1].s)
        al Rank'el('Tria 1].set_xlabs[0,
        axe')curaciese('Top 10 Ac1].set_titl0, es[        axcuracy'])
p_10['ac toen(top_10)),(range(ls[0, 1].bar        axeacies
ccur     # A   
   
     Score')Composite _ylabel('setxes[0, 0].    ank')
    l Raria_xlabel('Tetes[0, 0].sax')
        te Scoressi10 Compotle('Top , 0].set_ti[0        axesore'])
site_sc10['compo, top_op_10))range(len(t.bar([0, 0]      axess
  ite scoreosmp       # Co   
 ))
     15, 10size=(igs(2, 2, f.subplotaxes = plt     fig, 
     ')
      reposite_sco 'comgest(10,.nlartop_10 = dfn
        mparisols co0 tria 4. Top 1 #
           lose()
    .cplt')
        es='tightinch, bbox_ dpi=300g",rformance.pnsize_vs_pets_dir / "avefig(plot.s
        pl=0.3)(True, alphaid   plt.gr
     )Performance'ize vs tle('Model S.tiplt   %)')
     Accuracy (lidation 'Vael( plt.ylab
       ze (MB)')l('Model Sit.xlabe
        plScore')omposite er, label='Clorbar(scatt     plt.coa=0.7)
   lphplasma', aap='cme'], e_scormpositcodf['     c=                 , 
      ']['accuracydfze_mb'], _siodel'mter(df[atter = plt.sc     scat6))
   ze=(10, igsi.figure(f    pltrmance
    ze vs Perfo Si 3. Model #         
  ose()
    .cl     plt')
   'tightches=0, bbox_indpi=30e.png", uracy_vs_timir / "accig(plots_d   plt.savef)
     .3, alpha=0.grid(True     plt
   ng Time')rainiracy vs Ttitle('Accu    plt.
    acy (%)')n Accur'Validatiot.ylabel(pl')
        nds)secog Time (l('Traininlt.xlabe       p)
 ore'omposite Scr, label='Cscattelt.colorbar(     p   a=0.7)
 alphp='viridis',re'], cmaosite_sco  c=df['comp                     '], 
     curacy], df['ace'aining_tim'trcatter(df[.slt= per catt    s  , 6))
  e=(10figsizfigure(        plt.ime
aining Tvs Tr Accuracy    # 2.
          ose()
        plt.cl
   t')ighx_inches='t0, bbog", dpi=30ribution.pncore_dist"sts_dir / (plo plt.savefig       .3)
e, alpha=0ru(Trid   plt.g
     e Scores')ompositn of CDistributio plt.title('      y')
 equenc'Frylabel(    plt.)
    e Score'('Compositt.xlabel      plblack')
  or='7, edgecolpha=0., alns=30 bite_score'],composiist(df['  plt.h
      (10, 6))ze=ure(figsiplt.fig
        distributionosite score  Comp    # 1.  
        0_8')
  eaborn-v('s.useyle      plt.st
  Set style #     
           _results)
f.trialame(sel= pd.DataFrdf        
         return
       lts:
     _resurial.tif not self       
   
      """s plotsstom analysie cu """Creat    ):
   r: Pathplots_dielf, lots(se_custom_pf _creat    de  
: {e}")
  alizationsisureate viled to cf"Fager.warning( log         as e:
  t Exception   excep  
           }")
     iro {plots_ded tzations savisualiinfo(f"V    logger. 
                r)
   ts_dilolots(pate_custom_p_cre  self.
          atplotliblots using mCustom p   #          
        
    "))nce.htmlimportameter_/ "paras_dir plotr(rite_html(stfig.w            )
.studyrtances(selfm_impo plot_para      fig =ce
      rtanr impo  # Paramete             
         
.html"))n_historymizatio"optis_dir / plottml(str(ig.write_h  f
          f.study)y(seltion_histormizaoptilot_ig = p       fory
     zation histOptimi        # y:
      tr    
      True)
    exist_ok=ir.mkdir(ts_d plo"
       r / "plotssults_di self.rer =   plots_di     
          return
 
         ials:elf.study.trt sno   if   
       
    ""s"ontiisualiza vation optimiz""Create"      f):
  ions(selzatali_create_visu
    def ")
    s_dir}lf.resultsaved to {seults "Resger.info(f      log  
     
   e)x=Falsdes.csv", inult_resir / "trial.results_d_csv(self     df.to      
 lts)rial_resuFrame(self.tata   df = pd.D     s:
    trial_resultf self.
        imaryesults sum trial r Save   #
     
        t=2)gs, f, inden_confiston.dump(be         js      
 s f:on", 'w') aions.jsonfiguratr / "best_c.results_dith open(self       wi  
       
        nfig_dict)(coigs.append best_conf         ue']
      valal_info['core'] = tricomposite_sict['   config_d     
        r']mbefo['nu_iner'] = trialtrial_numbct['fig_dion c               _attrs'])
info['userte(trial_t.updadicconfig_              
  ].copy()rams'ial_info['patr= dict      config_        ]:
   p_trials'is['tofo in analysl_inia      for tr
       = []igsest_conf  b          als'):
rip_tet('to analysis.g if   ations
    t configur # Save bes 
              ndent=2)
ysis, f, ip(anal.dum  json
           f:", 'w') aslysis.json_anaptimizationr / "o_diself.results with open(      ysis
 nal# Save a        
     
   .study, f)(selfpickle.dump    
         'wb') as f:",_study.pklunadir / "optelf.results_en(swith op      
  udy stave
        # S       es"""
 o filsults tzation reimi""Save opt"       Any]):
  : Dict[str,ysiself, anals(sltization_resuptim_save_o  def is
    
  lysana     return    
        
}")ue']:.4fal['std_v{analysisf} ± lue']:.4_vas['mean: {analysioref"  Mean scfo( logger.in")
       f}e']:.4t_valulysis['bes: {anaore  Best scnfo(f" logger.i
       }")ls']d_triailealysis['fad: {anf"  Failegger.info(    lo")
    s']}ned_trialysis['pru: {anal"  Prunedo(flogger.inf   )
     ls']}"mpleted_tria['cosis {analy  Completed:o(f"gger.inflo        
s']}")total_trialis[': {analystrialsotal nfo(f"  Tger.i       logsis:")
 tion AnalymizaOpti(f"r.infoge  log  
         }
        
     ]         p_trials
  for t in to                       }
rs
        .user_att': tattrs'user_          
          rams,.pa: t   'params'           
      ': t.value,    'value            r,
    numbe': t.mber  'nu                          {
        s': [
trial    'top_       else 0,
 ) if values uesald(vnp.stue': d_valst      '    
   else 0,aluesalues) if vean(value': np.m_v  'mean         ,
 se 0 eluess) if valx(valuelue': ma_va       'best
     te.FAIL]),Starial.Trial== optuna.te .statials if tstudy.trself.n for t ien([t als': liled_tri    'fa     UNED]),
   alState.PRal.Trina.trite == optu t.statrials if.study.or t in selft fs': len([rial 'pruned_t         
  ls),pleted_trials': len(comtriampleted_        'co    s),
dy.trialf.stu: len(sels'l_trial       'tota
      {nalysis =        a        
s]
_trialn completedt it.value for  values = [ics
       istte statCalcula   #      
   ]
     _kg.save_topconfilf.[:seed_trialss = complet_trial       toprials
  top t     # Get     
   e)
   ruerse=Trevalue,  t: t.vey=lambda(k_trials.sortted   complee
     ctive valuby objetrials  # Sort 
       {}
        return       ")
      ls found tria completed"Noer.warning(     logg
       d_trials:ot complete     if n  
         COMPLETE]
.TrialState.trialna.= optuf t.state =udy.trials iin self.st = [t for t ed_trials    complets
     trialompleted     # Get c      
    {}
       return    :
   .trialstudyt self.s if no
       "
        esults""ization re optim"""Analyz
        tr, Any]:> Dict[self) -s(salyze_result  def _an
    
        }
  s_dir)lf.result(se str_dir':'results            otal_time,
ime': tion_toptimizat     'gs,
        best_confi':nfigsest_co         'brial,
   udy.best_tself.stl': ria_tst    'be,
        tudyself.study':     's  {
           return     
   s")
    second1f} me:.l_tid in {totaon completeptimizati.info(f"O      loggerime
  self.start_ttime() - ime.l_time = tta
        to)
        t_models(ain_bes= self._retrigs st_conf bels
       st mode berain      # Ret 
        )
 ns(atio_visualizelf._create        sts:
    t_resul.ploself.configf         iions
isualizatrate v    # Gene  
    
      ts(results)sul_reimizationsave_optlf._      se results
   Save       #       
 ts()
 esulanalyze_r = self._esults r  ults
     yze resnal A       #      
 r")
  ted by useinterrupon mizatiptiinfo("O     logger.
       pt:dInterruoarexcept Keyb)
                   =True
 rogress_bar show_p          
     _jobs,lf.config.nse     n_jobs=       ,
    imeoutelf.config.teout=stim                trials,
config.n_elf. n_trials=s        ,
       objective     self.          optimize(
 dy.    self.stu:
            try    on
optimizati      # Run    
  
     udy()eate_stcrf. = seltudy      self.s  e study
 # Creat 
           
   me()ime.ti= ttime  self.start_")
       ion...optimizateter paramhyperStarting nfo("  logger.i  """
    
        urations.ignfst co and ben resultsoptimizatio    Returns           
rocess.
  mization pr optimeterparapecomplete hy   Run the      "
      ""ny]:
   Dict[str, A ->f)e(self optimiz 
    des
   iled triale for fasible scorost pReturn wors  # turn 0.0     re     )
  k.print_exc(  tracebac        k
  cebacimport tra     )
       }"d: {eber} failel {trial.numiaror(f"Trr.er    logge   e:
      tion ascept Excep   ex     
         e_score
   urn compositret           
           
  .1f}MB")_mb:odel_size{m  f"Size:                      f}s, "
.1me:g_ti{trainin   f"Time:                   }%, "
  .2fal_accuracy:{best_v: acyf"Accur                  }, "
     _score:.4fsiteompo Score: {composite   f"C            
        leted: "mber} comptrial.nuf"Trial {nfo(logger.i    
                    ent=2)
lt, f, indrial_resu json.dump(t            as f:
    w')son", 'esults.jl.number}_rria_{t/ f"trialir s_dself.resulth open(         witults
   resrial Save t   #                  
 
   ult)esrial_rappend(ts.sultf.trial_re        sel    
             }
         g)
  dict(confi': as  'config              _loss,
 best_val_val_loss':      'best       mb,
   del_size_: moe_mb' 'model_siz        
       _time,ninge': training_timtrai     '
           l_accuracy,est_va by': 'accurac             score,
  composite__score': te'composi               r,
 rial.numbel_number': t     'tria    
       t = {al_resul    tri
        esultsrial r# Store t                
    )
               
 ze_score_weight * sisizeg.model_ self.confi     
          _score + * timeghte_wei_timingfig.trainf.conel       s        ore +
 uracy_sc * accweightg.accuracy_f.confi        sel        
te_score = ( composi           score
 ed compositeght# Wei              
          odels
ge mlare  # Penaliz_mb / 100)) el_size(0, 1 - (mode = max  size_scor     s
     ng timelong trainienalize # P / 3600))  _timeining(0, 1 - (traore = maxime_sc     t    
   o [0, 1]entage terc pConvert00.0  # racy / 1est_val_accuscore = b   accuracy_        e
  1] rang [0, metrics toalize      # Norm      ore
scsite mpococulate        # Cal   
        
      tart_time trial_s.time() -time = timeaining_     trime
       e training tulat    # Calc 
                   ss:.4f}")
{avg_val_loLoss: 2f}%, Val :.racyaccu{val_Val Acc:        f"                      h}: "
 h {epocer}, Epoc{trial.numbfo(f"Trial   logger.in                  0:
 10 == f epoch %          iess
       Log progr          #
                   break
                   er}")
    numbl {trial. tria forch} epoch {epong atstoppiy f"Earlo(r.inflogge            
        n_epochs):fig.miself.con>= h  epoc                   nd 
nce aopping_patiey_stearlig.onfelf.cnter >= satience_cou    if (p    
        stoppingEarly    #      
                   d()
     runeptuna.TrialP   raise o          ")
       och}t epoch {eped ar} prunrial.numberial {t(f"Togger.info l           
        une():ld_prou trial.sh         ifd
       rune should be peck if trial    # Ch      
                )
      epochuracy, val_accreport(rial.         t       ning
ue for prurmediate valort inte      # Rep          
               += 1
 er countience_    pat                :
se        el    
     = 0ternce_coun      patie       cy
       al_accuraaccuracy = vval_     best_            s
   l_losvg_vaal_loss = ast_vbe            s:
        val_losest_s < bval_los   if avg_           check
  ping Early stop#                 
              ()
  duler.step       sche          
   "plateau":ype != scheduler_t  if          
     e scheduler   # Updat           
                 oader)
 .val_l/ len(self val_loss _loss =   avg_val      r)
       in_loadetra(self.n_loss / lenrais = t_los_train         avg    l_total
   ect / va0 * val_corr= 100.y  val_accurac              
 train_totalrrect / * train_coacy = 100.0 urn_acc    trai          metrics
  ulate  # Calc                  
         )
    sum().item(ts).class_targeedicted ==  += (prrrect  val_co                  
    )ts.size(0_targeal += class   val_tot              )
       ta, 1its.das_logrch.max(clas = toedpredict      _,                  
 s.item()tal_los_loss += to         val              
                          reg_loss)
on_weight *sires config.reg                                 oss + 
   * class_ltion_weightsificaonfig.clasl_loss = (c    tota                  gets)
  ts, reg_tar_outpuSELoss()(regnn.M reg_loss =                        targets)
, class_ass_logitsopyLoss()(clossEntr = nn.Crass_loss   cl                       
                    
  l(data)d_moderitrainer.hyb _ = s,, reg_outputass_logits     cl                   
                    
    ice)evs.to(self.d= reg_targettargets reg_                      ice)
  to(self.devets. class_targets =_targ      class            ice)
      elf.devo(sdata.t =         data             ader:
   l_lon self.vaets i_targ, regrgets class_ta   for data,                :
 _grad()h.norc toith           w  
              l = 0
       val_tota            0
   orrect =val_c          0.0
      s = losl_     va          ()
 d_model.evalrier.hyb      train         ase
 alidation ph     # V
                      
     tem()s).sum().irget == class_tadictedre (pct +=train_corre                  size(0)
  s._targettal += classin_to      tra              data, 1)
_logits.x(classd = torch.mapredicte       _,             tem()
 tal_loss.i_loss += tointra           
         tatistics# S                      
                 p()
 ptimizer.ste      o          
    0)rs(), 1.arametebrid_model.piner.hynorm_(trap_grad_tils.cli torch.nn.u                ard()
   .backwl_loss tota                pass
     Backward     #              
                 _loss)
    ht * regeigregression_wnfig.co                           
     + class_loss weight * sification_.clasigconfloss = ( total_                 s)
  _targettputs, reg_ouss()(reg= nn.MSELo  reg_loss              
     rgets)ss_tas, claogit_llassss()(cpyLoossEntro= nn.Crs_loss         clas           
 sesculate los# Cal                   
           
          el(data)rid_modiner.hyb tra _ =tputs,reg_ou_logits, ass          cl        ss
  Forward pa         #                     
        _grad()
   ror.ze    optimize                 
              vice)
     .deselftargets.to(s = reg_reg_target                
    lf.device)(seargets.tos = class_target_tssla  c            ce)
      evilf.dta.to(seta = da      da              r):
adeain_lo(self.tr enumeratergets) ins, reg_ta_targetclass(data, tch_idx, ba    for          
               
    tal = 0n_to trai    
           ect = 0n_corr        trai     s = 0.0
   los      train_
          in()rabrid_model.tr.hyne