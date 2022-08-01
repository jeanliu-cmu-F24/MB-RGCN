import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from BaseDataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.test_epoch - (args.test_epoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.test_epoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.test_epoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def neumf(self, idx, ulat, ilat):
		lat1 = ulat * ilat
		lat2 = tf.concat([ulat, ilat], axis=-1)
		lat11 = tf.squeeze(FC(lat1, 1, reg=True, name='fc11_%d'%idx, reuse=True))
		lat21 = FC(lat2, args.latdim, useBias=True, reg=True, activation='relu', name='fc21_%d'%idx, reuse=True)
		lat22 = tf.squeeze(FC(lat21, 1, reg=True, name='fc22_%d'%idx, reuse=True))
		preds = (lat11 + lat22) * args.mult
		return preds

	def predict(self, tgt):
		allUEmbeds = NNs.getOrDefineParam('allUEmbeds', [args.user, args.latdim], reg=True, reuse=True)
		allIEmbeds = NNs.getOrDefineParam('allIEmbeds', [args.item, args.latdim], reg=True, reuse=True)
		uEmbeds = allUEmbeds
		iEmbeds = allIEmbeds

		preds = 0
		for idx in range(tgt+1):
			uids = self.uids[tgt]
			iids = self.iids[tgt]
			ulat = tf.nn.embedding_lookup(uEmbeds, uids)
			ilat = tf.nn.embedding_lookup(iEmbeds, iids)
			preds += self.neumf(idx, ulat, ilat)
		return preds

	def prepareModel(self):
		# NNs.leaky = 0.5
		self.actFunc = 'leakyRelu'
		self.uids, self.iids = [], []
		for i in range(args.behNum):
			self.uids.append(tf.placeholder(name='uids'+str(i), dtype=tf.int32, shape=[None]))
			self.iids.append(tf.placeholder(name='iids'+str(i), dtype=tf.int32, shape=[None]))
		
		self.preLoss = 0
		for tgt in range(args.behNum):
			preds = self.predict(tgt)
			sampNum = tf.shape(self.uids[tgt])[0] // 2
			posPred = tf.slice(preds, [0], [sampNum])
			negPred = tf.slice(preds, [sampNum], [-1])
			self.preLoss += tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
			if tgt == args.behNum - 1:
				self.targetPreds = preds
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

		# adam = tf.train.AdamOptimizer(learning_rate=learningRate)
		# gvs = adam.compute_gradients(self.loss)
		# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		# self.optimizer = adam.apply_gradients(capped_gvs)

	def sampleTrainBatch(self, batIds, labelMat, itmNum):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = []
				neglocs = []
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, itmNum)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		allIds = np.random.permutation(args.user)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		glbnum = len(allIds)

		bigSteps = int(np.ceil(glbnum / args.divSize))
		glb_i = 0
		glb_step = int(np.ceil(glbnum / args.batch))
		for s in range(bigSteps):
			bigSt = s * args.divSize
			bigEd = min((s+1) * args.divSize, glbnum)
			sfIds = allIds[bigSt: bigEd]
			num = bigEd - bigSt

			steps = num // args.batch

			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
				feed_dict = {}
				for beh in range(args.behNum):
					uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh], args.item)
					feed_dict[self.uids[beh]] = uLocs
					feed_dict[self.iids[beh]] = iLocs

				res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

				preLoss, regLoss, loss = res[1:]

				epochLoss += loss
				epochPreLoss += preLoss
				glb_i += 1
				log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (glb_i, glb_step, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / glb_step
		ret['preLoss'] = epochPreLoss / glb_step
		return ret

	def sampleTestBatch(self, batIds, labelMat, tstInt):
		batch = len(batIds)
		temTst = tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				cur += 1
		return uLocs, iLocs, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		allIds = self.handler.tstUsrs
		glbnum = len(allIds)
		tstBat = args.batch

		bigSteps = int(np.ceil(glbnum / args.divSize))
		glb_i = 0
		glb_step = int(np.ceil(glbnum / tstBat))
		for s in range(bigSteps):
			bigSt = s * args.divSize
			bigEd = min((s+1) * args.divSize, glbnum)
			ids = allIds[bigSt: bigEd]
			num = bigEd - bigSt

			steps = int(np.ceil(num / tstBat))

			for i in range(steps):
				st = i * tstBat
				ed = min((i+1) * tstBat, num)
				batIds = ids[st: ed]
				uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1], self.handler.tstInt)
				feed_dict = dict()
				feed_dict[self.uids[-1]] = uLocs
				feed_dict[self.iids[-1]] = iLocs
				preds = self.sess.run(self.targetPreds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
				hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
				epochHit += hit
				epochNdcg += ndcg
				glb_i += 1
				log('Steps %d/%d: hit = %d, ndcg = %d          ' % (glb_i, glb_step, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / glbnum
		ret['NDCG'] = epochNdcg / glbnum
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()