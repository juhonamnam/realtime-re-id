import { useCallback, useEffect, useRef, useState } from "react"
import { useCamData, Cam, CamWrapper } from "./cam"
import { logger } from "./logger"
import { useLoading } from "./loading"
import * as ort from "onnxruntime-web"
import { cropCanvasToTensor, videoToTensor } from "./convert"
import { Button, Modal } from "react-bootstrap"
import { Carousel } from "./carousel"
import type { Feature, FEModelInfo, PDModelInfo, Snap, Status } from "./type"
import {
  COLOR_OF_MATCH,
  COLOR_OF_UNMATCH,
  INITIAL_STATUS,
  LINE_WIDTH,
  FONT_SIZE,
} from "./const"
import {
  getSimilarityColor,
  getSimilarityScore,
  getTotalVisibilityColor,
  getVisibilityColor,
  relativelyVisible,
  visible,
} from "./func"

type ReIdentificationProps = {
  pdModel: PDModelInfo
  feModel: FEModelInfo
}

export const ReIdentification = ({
  pdModel,
  feModel,
}: ReIdentificationProps) => {
  const { setCamDataHandler, clear, flipRef, camRef } = useCamData()
  const { setLoading } = useLoading()

  const [session, setSession] = useState<{
    pd?: ort.InferenceSession
    fe?: ort.InferenceSession
  }>({})

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [status, setStatus] = useState<Status>(INITIAL_STATUS)

  const [snap, setSnap] = useState<Snap | null>(null)
  const featureToCompareRef = useRef<Feature | null>(null)

  const wrapperRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<HTMLDivElement>(null)

  const resizeInnerThrottleOccupied = useRef(false)

  useEffect(() => {
    if (wrapperRef.current === null || innerRef.current === null) return

    const wrapper = wrapperRef.current
    const inner = innerRef.current
    const cam = camRef.current

    const resizeInner = () => {
      if (resizeInnerThrottleOccupied.current) return

      resizeInnerThrottleOccupied.current = true
      setTimeout(() => {
        const camRatio = camRef.current.videoWidth / camRef.current.videoHeight

        const wrapperWidth = wrapper.clientWidth
        const wrapperHeight = wrapper.clientHeight

        const wrapperRatio = wrapperWidth / wrapperHeight

        if (camRatio > wrapperRatio) {
          inner.style.width = `${wrapperWidth}px`
          inner.style.height = `${wrapperWidth / camRatio}px`
        } else {
          inner.style.width = `${wrapperHeight * camRatio}px`
          inner.style.height = `${wrapperHeight}px`
        }

        resizeInnerThrottleOccupied.current = false
      }, 100)
    }

    const resizeObserverForInner = new ResizeObserver(resizeInner)
    resizeObserverForInner.observe(wrapper)
    resizeObserverForInner.observe(cam)
    cam.addEventListener("loadedmetadata", resizeInner)

    return () => {
      cam.removeEventListener("loadedmetadata", resizeInner)
      resizeObserverForInner.disconnect()
    }
  }, [camRef])

  const detectPerson = useCallback(
    async (
      pdSession: ort.InferenceSession,
      tensor: ort.Tensor,
      pdModel: PDModelInfo,
    ) => {
      const [inputName] = pdSession.inputNames
      const [outputName] = pdSession.outputNames

      const feeds = { [inputName]: tensor }
      const pdResult = await pdSession.run(feeds)

      const bboxes = pdResult[outputName]
      const bboxesData = bboxes.data as Float32Array

      const bboxList: [number, number, number, number][] = []

      const stride = bboxes.dims[2]

      for (let i = 0; i < bboxesData.length; i += stride) {
        const score = bboxesData[i + 4]
        if (score < pdModel.threshold) {
          continue
        }

        const x1 = bboxesData[i]
        const x2 = bboxesData[i + 2]
        const y1 = bboxesData[i + 1]
        const y2 = bboxesData[i + 3]

        bboxList.push([x1, y1, x2, y2])
      }
      bboxes.dispose()

      return bboxList
    },
    [],
  )

  const extractFeatures = useCallback(
    async (
      feSession: ort.InferenceSession,
      canvas: OffscreenCanvas,
      bboxes: [number, number, number, number][],
      feModel: FEModelInfo,
    ) => {
      const [inputName] = feSession.inputNames
      const [vScoreName, embVecName] = feSession.outputNames

      const pTensor = await cropCanvasToTensor(canvas, bboxes, feModel.shape)

      const feeds = { [inputName]: pTensor }
      const feResult = await feSession.run(feeds)

      pTensor.dispose()

      const embVec = feResult[embVecName]
      const embVecData = embVec.data as Float32Array

      const vScore = feResult[vScoreName]
      const vScoreData = vScore.data as Float32Array

      const batch = embVec.dims[0]
      const segments = embVec.dims[1]
      const featureNum = embVec.dims[2]

      embVec.dispose()
      vScore.dispose()

      const result: Feature[] = []

      for (let i = 0; i < batch; i++) {
        const embVecs: Float32Array[] = []
        const vScores: Float32Array = new Float32Array(segments)

        for (let j = 0; j < segments; j++) {
          const embVec = embVecData.slice(
            i * segments * featureNum + j * featureNum,
            i * segments * featureNum + (j + 1) * featureNum,
          )
          vScores[j] = vScoreData[i * segments + j]
          embVecs.push(embVec)
        }

        result.push({ embVecs, vScores })
      }

      return result
    },
    [],
  )

  const takeSnapshot = async (pdModel: PDModelInfo, feModel: FEModelInfo) => {
    if (!camRef.current) return
    if (!session.pd || !session.fe) return
    const { tensor, padding, canvas } = await videoToTensor(
      camRef.current,
      pdModel.shape,
    )
    const bboxes = await detectPerson(session.pd, tensor, pdModel)
    tensor.dispose()
    if (bboxes.length > 0) {
      const features = await extractFeatures(
        session.fe,
        canvas,
        bboxes,
        feModel,
      )
      setSnap({ canvas, bboxes, padding, features })
      setStatus("select")
    }
  }

  const predict = useCallback(
    async (
      camData: HTMLVideoElement,
      pdModel: PDModelInfo,
      feModel: FEModelInfo,
    ) => {
      if (!session.pd || !session.fe) return
      if (!canvasRef.current) return
      canvasRef.current.width = camData.clientWidth
      canvasRef.current.height = camData.clientHeight

      const ctx = canvasRef.current.getContext("2d")
      if (!ctx) return

      if (!featureToCompareRef.current) return

      const { tensor, padding, canvas } = await videoToTensor(
        camData,
        pdModel.shape,
      )

      const bboxes = await detectPerson(session.pd, tensor, pdModel)
      if (!bboxes.length) return

      const xRatio = camData.clientWidth / (pdModel.shape[1] * (1 - padding.x))
      const yRatio = camData.clientHeight / (pdModel.shape[0] * (1 - padding.y))

      const features = await extractFeatures(
        session.fe,
        canvas,
        bboxes,
        feModel,
      )

      for (let i = 0; i < bboxes.length; i++) {
        const [x1_, y1_, x2_, y2_] = bboxes[i]
        const feature = features[i]

        const similarityScore = getSimilarityScore(
          feature,
          featureToCompareRef.current,
          feModel.partSimilarityThresholds,
        )

        const isMatch = similarityScore >= feModel.similarityThreshold
        const isVisible = relativelyVisible(
          featureToCompareRef.current,
          feature,
          feModel.visibilityThreshold,
        )

        const matchColor =
          isMatch && isVisible ? COLOR_OF_MATCH : COLOR_OF_UNMATCH

        const x1 = flipRef.current
          ? camData.clientWidth - x2_ * xRatio
          : x1_ * xRatio
        const x2 = flipRef.current
          ? camData.clientWidth - x1_ * xRatio
          : x2_ * xRatio
        const y1 = y1_ * yRatio
        const y2 = y2_ * yRatio

        ctx.strokeStyle = `rgb(${matchColor.join(", ")})`
        ctx.lineWidth = LINE_WIDTH
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        ctx.font = `${FONT_SIZE}px Arial`

        const similarityColor = getSimilarityColor(
          similarityScore,
          feModel.similarityThreshold,
        )

        const similarityText = `Similarity: ${similarityScore.toFixed(2)}`

        const visibilityColors = [
          getTotalVisibilityColor(feature, feModel.visibilityThreshold),
        ]

        const visibilityTexts = ["Visibility: "]

        for (let j = 0; j < feature.vScores.length; j++) {
          visibilityColors.push(
            getVisibilityColor(feature.vScores[j], feModel.visibilityThreshold),
          )
          visibilityTexts.push(feature.vScores[j].toFixed(2) + " ")
        }

        const textWidth = Math.max(
          ctx.measureText(similarityText).width,
          ctx.measureText(visibilityTexts.join("")).width,
        )

        ctx.fillStyle = "rgba(0, 0, 0, 0.5)"
        ctx.fillRect(
          x1,
          Math.max(y1 - 5 - FONT_SIZE * 2, 0),
          textWidth + LINE_WIDTH * 2,
          FONT_SIZE * 2 + 5,
        )

        const similarityTextY = Math.max(y1 - 5 - FONT_SIZE, FONT_SIZE)

        ctx.fillStyle = `rgb(${similarityColor.join(", ")})`
        ctx.fillText(similarityText, x1 + LINE_WIDTH, similarityTextY)

        let visibilityTextX = x1 + LINE_WIDTH

        for (let j = 0; j < visibilityColors.length; j++) {
          const visibilityColor = visibilityColors[j]
          const visibilityText = visibilityTexts[j]
          ctx.fillStyle = `rgb(${visibilityColor.join(", ")})`
          ctx.fillText(
            visibilityText,
            visibilityTextX,
            similarityTextY + FONT_SIZE,
          )

          visibilityTextX += ctx.measureText(visibilityText).width
        }
      }
    },
    [flipRef, session, detectPerson, extractFeatures],
  )

  const onSelect = useCallback(
    (idx: number, pdModel: PDModelInfo, feModel: FEModelInfo) => {
      if (!snap) return
      featureToCompareRef.current = snap.features[idx]
      setStatus("reid")
      setCamDataHandler((camData) => predict(camData, pdModel, feModel))
    },
    [snap, setCamDataHandler, predict],
  )
  const onClose = () => {
    setStatus("default")
  }

  const reset = () => {
    clear()
    setStatus("default")
  }

  useEffect(() => {
    setStatus(INITIAL_STATUS)
    logger("Loading Start")
    setLoading(true)
    const loadModel = Promise.all([
      ort.InferenceSession.create(pdModel.path),
      ort.InferenceSession.create(feModel.path),
    ])
      .then(([pdSession, feSession]) => {
        setSession({ pd: pdSession, fe: feSession })
        logger("Loading Finished")
        setLoading(false)
        return [pdSession, feSession]
      })
      .catch((reason) => {
        alert(reason)
        setLoading(false)
        return []
      })
    return () => {
      loadModel.then(([pdSession, feSession]) => {
        logger("Unloaded")
        pdSession?.release()
        feSession?.release()
        clear()
      })
    }
  }, [clear, setLoading, pdModel, feModel])

  return (
    <>
      <div className="camera-container" ref={wrapperRef}>
        <div ref={innerRef}>
          <CamWrapper>
            <Cam />
            <canvas
              className={`position-absolute end-0 top-0${status === "reid" ? "" : " d-none"}`}
              ref={canvasRef}
            />
          </CamWrapper>
        </div>
      </div>
      <div className="camera-button-container z-2">
        {status === "default" && (
          <button
            className="camera-button bi bi-record-circle"
            onClick={() => {
              if (!pdModel || !feModel) return
              takeSnapshot(pdModel, feModel)
            }}
          />
        )}
        {status === "reid" && (
          <button
            className="camera-button bi bi-arrow-clockwise"
            onClick={() => {
              reset()
            }}
          ></button>
        )}
      </div>
      {feModel && (
        <PersonSelectModal
          show={status === "select"}
          snap={snap}
          onSelect={(idx) => {
            if (!pdModel || !feModel) return
            onSelect(idx, pdModel, feModel)
          }}
          onClose={onClose}
          feModel={feModel}
        />
      )}
    </>
  )
}

const PersonSelectModal = ({
  show,
  snap,
  onSelect,
  onClose,
  feModel,
}: {
  show: boolean
  snap: Snap | null
  onSelect: (idx: number) => void
  onClose: () => void
  feModel: FEModelInfo
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const canvasWrapperRef = useRef<HTMLDivElement>(null)
  const [cropUrls, setCropUrls] = useState<string[]>([])
  const slideRef = useRef(0)
  const [slide, setSlide] = useState(0)

  const resizeThrottleOccupied = useRef(false)

  useEffect(() => {
    if (!snap || !canvasRef.current || !canvasWrapperRef.current) return
    const ctx = canvasRef.current.getContext("2d")
    if (!ctx) return
    ;(async () => {
      const canvasForPerson = new OffscreenCanvas(0, 0)
      const ctxForPerson = canvasForPerson.getContext("2d")
      if (!ctxForPerson) return

      const cropUrls_: string[] = []

      for (const bbox of snap.bboxes) {
        const [x1, y1, x2, y2] = bbox
        const width = x2 - x1
        const height = y2 - y1

        canvasForPerson.width = width
        canvasForPerson.height = height

        ctxForPerson.drawImage(
          snap.canvas,
          x1,
          y1,
          width,
          height,
          0,
          0,
          width,
          height,
        )

        const blob = await canvasForPerson.convertToBlob()
        const url = URL.createObjectURL(blob)
        cropUrls_.push(url)
      }

      setSlide(0)
      slideRef.current = 0
      setCropUrls(cropUrls_)
    })()

    const sourceWidth = snap.canvas.width * (1 - snap.padding.x)
    const sourceHeight = snap.canvas.height * (1 - snap.padding.y)

    const canvasForScene = new OffscreenCanvas(sourceWidth, sourceHeight)
    const ctxForScene = canvasForScene.getContext("2d")
    if (!ctxForScene) return

    ctxForScene.drawImage(
      snap.canvas,
      0,
      0,
      sourceWidth,
      sourceHeight,
      0,
      0,
      sourceWidth,
      sourceHeight,
    )

    for (let i = 0; i < snap.bboxes.length; i++) {
      const feature = snap.features[i]
      const [x1, y1, x2, y2] = snap.bboxes[i]

      const visibleColor = visible(feature, feModel.visibilityThreshold)
        ? COLOR_OF_MATCH
        : COLOR_OF_UNMATCH

      ctxForScene.strokeStyle = `rgb(${visibleColor.join(", ")}`
      ctxForScene.lineWidth = LINE_WIDTH
      ctxForScene.strokeRect(x1, y1, x2 - x1, y2 - y1)
    }

    const resizeObserver = new ResizeObserver((entries) => {
      if (resizeThrottleOccupied.current) return

      resizeThrottleOccupied.current = true
      setTimeout(() => {
        const targetWidth = entries[0].contentBoxSize[0].inlineSize
        const targetHeight = targetWidth / (sourceWidth / sourceHeight)

        ctx.canvas.width = targetWidth
        ctx.canvas.height = targetHeight

        ctx.drawImage(canvasForScene, 0, 0, targetWidth, targetHeight)
        resizeThrottleOccupied.current = false
      }, 300)
    })

    resizeObserver.observe(canvasWrapperRef.current)

    return () => {
      resizeObserver.disconnect()
    }
  }, [snap, feModel])

  return (
    <Modal show={show} onHide={onClose} keyboard={false} backdrop="static">
      <Modal.Header closeButton>
        <Modal.Title>Select Person</Modal.Title>
      </Modal.Header>
      <Modal.Body ref={canvasWrapperRef}>
        <canvas ref={canvasRef} />
        <div>
          <Carousel
            imageUrls={cropUrls}
            descriptions={snap?.features.map(
              ({ vScores }) =>
                `Visibility: ${Array.from(vScores)
                  .map((score) => score.toFixed(2))
                  .join(" ")}`,
            )}
            aspectRatio={1.4}
            slideRef={slideRef}
            slideState={[slide, setSlide]}
          />
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="primary" onClick={() => onSelect(slideRef.current)}>
          Select
        </Button>
        <Button variant="secondary" onClick={() => onClose()}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  )
}
