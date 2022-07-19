import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler2 import negSamp, transpose, DataHandler, transToLsts
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

	def messagePropagate(self, lat, mats, rowNums):		
		catlat1 = []
		wt = [None] * args.behNum		
		alpha = [None] * args.behNum
		wMultiN = [None] * args.behNum
		for b in range(args.behNum):
			wt[b] = NNs.defineRandomNameParam([1], reg=True)
		wt = tf.nn.softmax(wt, axis=1)

		allNums = tf.add_n(rowNums)
		for b in range(args.behNum):
			wMultiN[b] = wt[b] * rowNums[b]
		allMultix = tf.add_n(wMultiN)
		for b in range(args.behNum):
			alpha[b] = wMultiN[b] / (allMultix + 1e-8)

		for b in range(args.behNum):
			tem = alpha[b] * mats[b]
			temlat = tf.sparse.sparse_dense_matmul(tem, lat)
			transLat = FC(temlat, args.latdim, reg=True, useBias=False, activation='twoWayLeakyRelu6')
			catlat1.append(transLat)
			
		lat = tf.reduce_sum(tf.stack(catlat1, axis=-1), axis=-1)
		return lat


	def ours(self):
		all_uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		all_iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uEmbed0 = all_uEmbed0
		iEmbed0 = all_iEmbed0
		ulats = [uEmbed0]
		ilats = [iEmbed0]
		
		for i in range(args.gnn_layer):
			ulat = self.messagePropagate(ilats[-1], self.uiMats, self.uirowsum)
			ilat = self.messagePropagate(ulats[-1], self.iuMats, self.iurowsum)
			ulats.append(ulat + ulats[-1])
			ilats.append(ilat + ilats[-1])
		
		for i in range(args.gnn_layer+1):
			ulats[i] = ulats[i] / (1e-6+tf.sqrt(1e-6+tf.reduce_sum(tf.square(ulats[i]), axis=-1, keepdims=True)))
			ilats[i] = ilats[i] / (1e-6+tf.sqrt(1e-6+tf.reduce_sum(tf.square(ilats[i]), axis=-1, keepdims=True)))

		UEmbedPred = NNs.defineParam('UEmbedPred', shape=[args.user, args.latdim], dtype=tf.float32, reg=False)
		IEmbedPred = NNs.defineParam('IEmbedPred', shape=[args.item, args.latdim], dtype=tf.float32, reg=False)
		ulats[0] = UEmbedPred#tf.nn.embedding_lookup(UEmbedPred, self.all_usrs)
		ilats[0] = IEmbedPred#tf.nn.embedding_lookup(IEmbedPred, self.all_itms)
		
		# ulat = tf.concat(ulats, axis=-1)
		# ilat = tf.concat(ilats, axis=-1)
		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)
		pckULat = tf.nn.embedding_lookup(ulat, self.uids)
		pckILat = tf.nn.embedding_lookup(ilat, self.iids)
		predLat = pckULat * pckILat * args.mult
		pred = tf.squeeze(tf.reduce_sum(predLat, axis=1))
		return pred

	def prepareModel(self):
		self.uiMats = []
		self.iuMats = []
		self.uirowsum = []
		self.iurowsum = []

		for b in range(args.behNum):
			self.uiMats.append(tf.sparse_placeholder(dtype=tf.float32))
			self.iuMats.append(tf.sparse_placeholder(dtype=tf.float32))
			self.uirowsum.append(tf.placeholder(dtype=tf.float32, shape=[None, 1]))
			self.iurowsum.append(tf.placeholder(dtype=tf.float32, shape=[None, 1]))
		
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

		self.preds = self.ours()
		sampNum = tf.shape(self.iids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])
		negPred = tf.slice(self.preds, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

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

			feed_dict = {}
			for i in range(args.behNum):
				tpadj = transpose(self.handler.trnMats[i])
				feed_dict[self.uiMats[i]] = transToLsts(self.handler.trnMats[i], norm=True)
				feed_dict[self.iuMats[i]] = transToLsts(tpadj, norm=True)
				feed_dict[self.uirowsum[i]] = np.array(self.handler.trnMats[i].sum(1)).reshape([-1, 1])
				feed_dict[self.iurowsum[i]] = np.array(tpadj.sum(1)).reshape([-1, 1])

			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
				uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[-1], args.item)
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs

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

			
			feed_dict = {}
			for i in range(args.behNum):
				tpadj = transpose(self.handler.trnMats[i])
				feed_dict[self.uiMats[i]] = transToLsts(self.handler.trnMats[i], norm=True)
				feed_dict[self.iuMats[i]] = transToLsts(tpadj, norm=True)
				feed_dict[self.uirowsum[i]] = np.array(self.handler.trnMats[i].sum(1)).reshape([-1, 1])
				feed_dict[self.iurowsum[i]] = np.array(tpadj.sum(1)).reshape([-1, 1])

			for i in range(steps):
				st = i * tstBat
				ed = min((i+1) * tstBat, num)
				batIds = ids[st: ed]
				uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1], self.handler.tstInt)
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs
				preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
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