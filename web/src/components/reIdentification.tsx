import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { useCamData, Cam, CamWrapper } from "./cam"
import { logger } from "./logger"
import { useLoading } from "./loading"
import { cropCanvasToBuffer, captureVideo } from "./convert"
import { Button, Modal } from "react-bootstrap"
import { Carousel } from "./carousel"
import type { Feature, Snap, Status } from "./type"
import { LINE_WIDTH, FONT_SIZE } from "./const"
import {
  COLOR_OF_MATCH,
  COLOR_OF_UNMATCH,
  getSimilarityColor,
  getVisibilityColor,
  mergeColors,
} from "./color"

import {
  useReIDConfig,
  type PDModelInfo,
  type FEModelInfo,
  METRIC_TYPES,
} from "./reidConfig"

import { reidWorker } from "./worker"

const getTotalVisibilityScore = (feature: Feature) => {
  return (
    feature.vScores.reduce((acc, score) => acc + score, 0) /
    feature.vScores.length
  )
}

const getComparability = (featureToCompare: Feature, feature: Feature) => {
  return (
    feature.vScores.reduce((acc, score, i) => {
      const compareScore = featureToCompare.vScores[i]
      const unknownRatio = 1 - compareScore

      return (
        acc +
        unknownRatio +
        compareScore * (score / Math.max(score, compareScore))
      )
    }) / feature.vScores.length
  )
}

export const ReIdentification = () => {
  const { setLoading } = useLoading()
  const { setCamDataHandler, flip, camRef } = useCamData()
  const { pdModel, feModel, metricType, showComparsionDetail } = useReIDConfig()

  const flipRef = useRef(false)
  const metricTypeRef = useRef(METRIC_TYPES[0])
  const comparisonDetailRef = useRef(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [status, setStatus] = useState<Status>("default")

  const [snap, setSnap] = useState<Snap | null>(null)
  const featureToCompareRef = useRef<Feature | null>(null)

  const wrapperRef = useRef<HTMLDivElement>(null)
  const innerRef = useRef<HTMLDivElement>(null)

  const resizeInnerThrottleOccupied = useRef(false)

  useEffect(() => {
    flipRef.current = flip
  }, [flip])

  useEffect(() => {
    metricTypeRef.current = metricType
  }, [metricType])

  useEffect(() => {
    comparisonDetailRef.current = showComparsionDetail
  }, [showComparsionDetail])

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

  const takeSnapshot = async (pdModel: PDModelInfo, feModel: FEModelInfo) => {
    if (!camRef.current) return
    const {
      data: pdData,
      dims: pdDims,
      canvas,
      resizeScale,
    } = await captureVideo(camRef.current, pdModel.shape)

    const bboxes = await reidWorker.detect({
      input: { data: pdData, dims: pdDims, resizeScale },
      threshold: pdModel.threshold,
    })

    if (bboxes.length > 0) {
      // Need to extract features for these bboxes
      const { data: feData, dims: feDims } = await cropCanvasToBuffer(
        canvas,
        bboxes,
        feModel.shape,
      )
      const features = await reidWorker.extractFeatures({
        input: { data: feData, dims: feDims },
      })

      setSnap({ canvas, bboxes, features })
      setStatus("select")
    }
  }

  const predict = useCallback(
    async (
      camData: HTMLVideoElement,
      pdModel: PDModelInfo,
      feModel: FEModelInfo,
    ) => {
      if (!featureToCompareRef.current) return

      const {
        data: pdData,
        dims: pdDims,
        canvas,
        resizeScale,
      } = await captureVideo(camData, pdModel.shape)

      const bboxes = await reidWorker.detect({
        input: { data: pdData, dims: pdDims, resizeScale },
        threshold: pdModel.threshold,
      })

      const { data: feData, dims: feDims } = await cropCanvasToBuffer(
        canvas,
        bboxes,
        feModel.shape,
      )

      const features = await reidWorker.extractFeatures({
        input: { data: feData, dims: feDims },
      })

      const xRatio = camData.clientWidth / canvas.width
      const yRatio = camData.clientHeight / canvas.height

      if (!canvasRef.current) return

      const ctx = canvasRef.current.getContext("2d")
      if (!ctx) return

      canvasRef.current.width = camData.clientWidth
      canvasRef.current.height = camData.clientHeight

      for (let i = 0; i < bboxes.length; i++) {
        const [x1_, y1_, x2_, y2_] = bboxes[i]
        const feature = features[i]

        const [totalSScore, partSScores] = metricTypeRef.current.function(
          feature,
          featureToCompareRef.current,
          feModel.partSimilarityThresholds,
        )
        const totalVScore = getTotalVisibilityScore(feature)

        const isMatch =
          totalSScore >=
          feModel.similarityThresholds[metricTypeRef.current.name]

        const comparability = getComparability(
          featureToCompareRef.current,
          feature,
        )

        const comparable = comparability >= feModel.visibilityThreshold

        const matchColor =
          isMatch && comparable ? COLOR_OF_MATCH : COLOR_OF_UNMATCH

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

        if (comparisonDetailRef.current) {
          ctx.font = `${FONT_SIZE}px Arial`

          const totalSimilarityColor = getSimilarityColor(
            totalSScore,
            feModel.similarityThresholds[metricTypeRef.current.name],
          )

          const totalSimilarityText = `Total Similarity: ${totalSScore.toFixed(2)}`

          const totalVisibilityColor = getVisibilityColor(
            totalVScore,
            feModel.visibilityThreshold,
          )

          const totalVisibilityText = `Total Visibility: ${totalVScore.toFixed(2)}`

          const segmentNameTexts = feModel.segmentNames.map(
            (name) => name + " ",
          )

          const similarityTexts = ["Similarity "]

          const similarityColors = [totalSimilarityColor]

          for (let j = 0; j < feature.vScores.length; j++) {
            similarityColors.push(
              getSimilarityColor(
                partSScores[j],
                feModel.partSimilarityThresholds[j],
              ),
            )
            similarityTexts.push(partSScores[j].toFixed(2) + " ")
          }

          const visibilityColors = [totalVisibilityColor]

          const visibilityTexts = ["Visibility "]

          for (let j = 0; j < feature.vScores.length; j++) {
            visibilityColors.push(
              getVisibilityColor(
                feature.vScores[j],
                feModel.visibilityThreshold,
              ),
            )
            visibilityTexts.push(feature.vScores[j].toFixed(2) + " ")
          }

          const segmentNameColors = []

          for (let j = 0; j < feature.vScores.length; j++) {
            segmentNameColors.push(
              mergeColors(similarityColors[j + 1], visibilityColors[j + 1]),
            )
          }

          const tableWidths = [
            Math.max(
              ...segmentNameTexts.map((text) => ctx.measureText(text).width),
            ),
            Math.max(
              ...similarityTexts.map((text) => ctx.measureText(text).width),
            ),
            Math.max(
              ...visibilityTexts.map((text) => ctx.measureText(text).width),
            ),
          ]

          const textWidth = Math.max(
            ctx.measureText(totalSimilarityText).width,
            ctx.measureText(totalVisibilityText).width,
            tableWidths.reduce((acc, cur) => acc + cur, 0),
          )

          ctx.fillStyle = "rgba(0, 0, 0, 0.5)"
          ctx.fillRect(
            x1,
            Math.max(y1 - 5 - FONT_SIZE * (3 + feModel.segmentNames.length), 0),
            textWidth + LINE_WIDTH * 2,
            FONT_SIZE * (3 + feModel.segmentNames.length) + 5,
          )

          const topY = Math.max(
            y1 - 5 - FONT_SIZE * (2 + feModel.segmentNames.length),
            FONT_SIZE,
          )

          ctx.fillStyle = `rgb(${totalSimilarityColor.join(", ")})`
          ctx.fillText(totalSimilarityText, x1 + LINE_WIDTH, topY)

          ctx.fillStyle = `rgb(${totalVisibilityColor.join(", ")})`
          ctx.fillText(totalVisibilityText, x1 + LINE_WIDTH, topY + FONT_SIZE)

          for (let j = 0; j < similarityTexts.length; j++) {
            if (j !== 0) {
              ctx.fillStyle = `rgb(${segmentNameColors[j - 1].join(", ")})`
              ctx.fillText(
                segmentNameTexts[j - 1],
                x1 + LINE_WIDTH,
                topY + FONT_SIZE * 2 + j * FONT_SIZE,
              )
            }

            const similarityColor = similarityColors[j]
            const similarityText = similarityTexts[j]
            ctx.fillStyle = `rgb(${similarityColor.join(", ")})`
            ctx.fillText(
              similarityText,
              x1 + LINE_WIDTH + tableWidths[0],
              topY + FONT_SIZE * 2 + j * FONT_SIZE,
            )

            const visibilityColor = visibilityColors[j]
            const visibilityText = visibilityTexts[j]
            ctx.fillStyle = `rgb(${visibilityColor.join(", ")})`
            ctx.fillText(
              visibilityText,
              x1 + LINE_WIDTH + tableWidths[0] + tableWidths[1],
              topY + FONT_SIZE * 2 + j * FONT_SIZE,
            )
          }
        } else {
          ctx.font = `${FONT_SIZE}px Arial`
          ctx.fillStyle = `rgb(${matchColor.join(", ")})`
          ctx.fillText(totalSScore.toFixed(2), x1 + LINE_WIDTH, y1 - 5)
        }
      }
    },
    [flipRef],
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
    setCamDataHandler(null)
    setStatus("default")
  }

  useEffect(() => {
    setStatus("default")
    logger("Loading Start")
    setLoading(true)

    reidWorker
      .loadModels(pdModel.path, feModel.path)
      .then(() => {
        logger("Loading Finished")
        setLoading(false)
      })
      .catch((reason) => {
        alert(reason)
        setLoading(false)
      })

    return () => {
      setCamDataHandler(null)
    }
  }, [setCamDataHandler, setLoading, pdModel, feModel])

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

  const partVisibilityColors = useMemo(() => {
    if (!snap) return []
    return snap.features.map((feature) =>
      Array.from(feature.vScores).map((score) =>
        getVisibilityColor(score, feModel.visibilityThreshold),
      ),
    )
  }, [snap, feModel])

  const totalVisibilities = useMemo(() => {
    if (!snap) return []
    return snap.features.map((feature) => {
      const visibilityScore = getTotalVisibilityScore(feature)
      return [
        visibilityScore,
        getVisibilityColor(visibilityScore, feModel.visibilityThreshold),
      ] as const
    })
  }, [snap, feModel])

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

    const sourceWidth = snap.canvas.width
    const sourceHeight = snap.canvas.height

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

      const visibleColor =
        getTotalVisibilityScore(feature) >= feModel.visibilityThreshold
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
            descriptionLines={feModel.segmentNames.length + 1}
            descriptions={snap?.features.map(({ vScores }, i) => (
              <Fragment key={i}>
                <table>
                  <tbody>
                    <tr>
                      <td>Total Visibility:</td>
                      <td
                        style={{
                          color: `rgb(${totalVisibilities[i][1].join(", ")})`,
                        }}
                      >
                        {totalVisibilities[i][0].toFixed(2)}
                      </td>
                    </tr>
                    {Array.from(vScores).map((score, j) => (
                      <tr key={j}>
                        <td>{feModel.segmentNames[j]}:</td>
                        <td
                          style={{
                            color: `rgb(${partVisibilityColors[i][j].join(", ")})`,
                          }}
                        >
                          {score.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Fragment>
            ))}
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
