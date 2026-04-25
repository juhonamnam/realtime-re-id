import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { logger } from "../logger"

const DRAG_SENSITIVITY = 15

type Status =
  | "stationary"
  | "clicked"
  | "clickCanceled"
  | "dragging"
  | "dragEnding"
  | "moving-left"
  | "moving-right"

type DragOption = {
  startingClientX: number
  startingClientY: number
  currentTranslateX: number
}

type ClickMove = "left" | "right" | null

export const Carousel = ({
  imageUrls,
  descriptions,
  slideRef,
  slideState: [slide, _setSlide],
  aspectRatio = 1,
}: {
  imageUrls: string[]
  descriptions?: React.ReactNode[]
  slideRef: React.RefObject<number>
  slideState: [number, React.Dispatch<React.SetStateAction<number>>]
  aspectRatio?: number
}) => {
  const carouselRef = useRef<HTMLDivElement>(
    null,
  ) as React.RefObject<HTMLDivElement>

  const setSlide = useCallback(
    (slide: number) => {
      _setSlide(slide)
      slideRef.current = slide
    },
    [_setSlide, slideRef],
  )

  const [status, _setStatus] = useState<Status>("stationary")
  const statusRef = useRef<Status>("stationary")
  const setStatus = (status: Status) => {
    _setStatus(status)
    statusRef.current = status
  }

  const [dragOption, _setDragOption] = useState<DragOption>({
    startingClientX: 0,
    startingClientY: 0,
    currentTranslateX: 0,
  })
  const dragOptionRef = useRef<DragOption>({
    startingClientX: 0,
    startingClientY: 0,
    currentTranslateX: 0,
  })
  const setDragOption = (dragOption: DragOption) => {
    _setDragOption(dragOption)
    dragOptionRef.current = dragOption
  }

  const [moveOption, setMoveOption] = useState({
    srcIdx: 0,
    dstIdx: 0,
  })

  const clickMoveRef = useRef<ClickMove>(null)
  const setClickMove = (clickMove: ClickMove) => {
    clickMoveRef.current = clickMove
  }

  useEffect(() => {
    logger("carousel:", status)
  }, [status])

  const click = (
    status: Status,
    clientX: number,
    clientY: number,
    carouselWidth: number,
  ) => {
    if (status !== "stationary") return
    setDragOption({
      startingClientX: clientX,
      startingClientY: clientY,
      currentTranslateX: -carouselWidth,
    })
    setStatus("clicked")
  }

  const dragging = useCallback(
    (dragOption: DragOption, clientX: number, carouselWidth: number) => {
      let moveTranslateX = clientX - dragOption.startingClientX

      if (moveTranslateX > carouselWidth) {
        moveTranslateX = carouselWidth
      } else if (moveTranslateX < -carouselWidth) {
        moveTranslateX = -carouselWidth
      }

      setDragOption({
        ...dragOption,
        currentTranslateX: moveTranslateX - carouselWidth,
      })
    },
    [],
  )

  const dragEnd = useCallback(
    (slide: number, dragOption: DragOption, carouselWidth: number) => {
      let move = 0
      if (dragOption.currentTranslateX < -carouselWidth * 1.1) {
        move = 1
      } else if (dragOption.currentTranslateX > -carouselWidth * 0.9) {
        move = -1
      }

      setDragOption({
        ...dragOption,
        currentTranslateX: -carouselWidth * (move + 1),
      })

      setStatus("dragEnding")

      setTimeout(() => {
        setDragOption({
          ...dragOption,
          currentTranslateX: -carouselWidth,
        })
        setStatus("stationary")
        setSlide((slide + move + imageUrls.length) % imageUrls.length)
      }, 300)
    },
    [imageUrls, setSlide],
  )

  const move = useCallback(
    (srcIdx: number, dstIdx: number) => {
      setSlide(dstIdx)
      if (srcIdx < dstIdx) {
        setStatus("moving-right")
      } else {
        setStatus("moving-left")
      }

      setMoveOption({ srcIdx, dstIdx })

      setTimeout(() => {
        setClickMove(null)
        setStatus("stationary")
      }, 300)
    },
    [setSlide],
  )

  /* Events */
  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      const status = statusRef.current

      if (status === "clicked") {
        setStatus("dragging")
      } else if (status === "dragging") {
        e.preventDefault()
        dragging(
          dragOptionRef.current,
          e.clientX,
          carouselRef.current.clientWidth,
        )
      }
    },
    [dragging],
  )

  const onTouchMove = useCallback(
    (e: TouchEvent) => {
      const status = statusRef.current

      if (status === "clicked") {
        e.preventDefault()
        const xMove =
          e.targetTouches[0].clientX - dragOptionRef.current.startingClientX
        const yMove =
          e.targetTouches[0].clientY - dragOptionRef.current.startingClientY
        if (Math.abs(xMove) > DRAG_SENSITIVITY) {
          setStatus("dragging")
        } else if (Math.abs(yMove) > DRAG_SENSITIVITY) {
          setStatus("clickCanceled")
        }
      } else if (status === "dragging") {
        e.preventDefault()
        dragging(
          dragOptionRef.current,
          e.targetTouches[0].clientX,
          carouselRef.current.clientWidth,
        )
      }
    },
    [dragging],
  )

  const onMouseTouchUp = useCallback(() => {
    const status = statusRef.current
    const clickMove = clickMoveRef.current
    const slide = slideRef.current

    if (status === "clicked") {
      if (clickMove === "left") {
        move(slide, (slide + imageUrls.length - 1) % imageUrls.length)
      } else if (clickMove === "right") {
        move(slide, (slide + 1) % imageUrls.length)
      } else {
        setStatus("stationary")
      }
    } else if (status === "dragging") {
      dragEnd(slide, dragOptionRef.current, carouselRef.current.clientWidth)
    } else if (status === "clickCanceled") {
      setStatus("stationary")
    }
  }, [dragEnd, move, imageUrls, slideRef])

  useEffect(() => {
    const carouselElement = carouselRef.current

    window.addEventListener("mousemove", onMouseMove)
    carouselElement.addEventListener("touchmove", onTouchMove)
    window.addEventListener("mouseup", onMouseTouchUp)
    window.addEventListener("touchend", onMouseTouchUp)
    return () => {
      window.removeEventListener("mousemove", onMouseMove)
      carouselElement.removeEventListener("touchmove", onTouchMove)
      window.removeEventListener("mouseup", onMouseTouchUp)
      window.removeEventListener("touchend", onMouseTouchUp)
    }
  }, [onMouseMove, onTouchMove, onMouseTouchUp])

  const onIndicatorClick = useCallback(
    (status: Status, srcIdx: number, dstIdx: number) => {
      if (status !== "stationary" || srcIdx === dstIdx) return
      move(srcIdx, dstIdx)
    },
    [move],
  )

  const carouselListDisplay = useMemo(() => {
    if (imageUrls.length === 0) return []
    const carouselItems = imageUrls.map((url, idx) => ({ url, idx }))

    let carouselItemFiltered: { url: string; idx: number }[]
    if (imageUrls.length >= 2) {
      if (["dragging", "dragEnding"].includes(status)) {
        carouselItemFiltered = [
          ...(slide === 0
            ? imageUrls.length === 2
              ? [{ ...carouselItems[1], idx: -1 }]
              : [carouselItems[carouselItems.length - 1]]
            : []),
          ...carouselItems.slice(slide === 0 ? 0 : slide - 1, slide + 2),
          ...(slide === carouselItems.length - 1
            ? imageUrls.length === 2
              ? [{ ...carouselItems[0], idx: 2 }]
              : [carouselItems[0]]
            : []),
        ]
      } else if (status === "moving-right") {
        carouselItemFiltered = carouselItems.slice(
          moveOption.srcIdx,
          moveOption.dstIdx + 1,
        )
      } else if (status === "moving-left") {
        carouselItemFiltered = carouselItems.slice(
          moveOption.dstIdx,
          moveOption.srcIdx + 1,
        )
      } else {
        carouselItemFiltered = [carouselItems[slide]]
      }
    } else {
      carouselItemFiltered = carouselItems
    }

    let style = document.getElementById("my-carousel-transition")
    if (!style) {
      style = document.createElement("style")
      style.id = "my-carousel-transition"
      document.head.appendChild(style)
    }

    style.innerHTML = `
      @keyframes my-carousel-moving-right {
        from {
          transform: translateX(0);
        }
        to {
          transform: translateX(${100 / carouselItemFiltered.length - 100}%);
        }
      }`

    const width = `${100 / carouselItemFiltered.length}%`

    return carouselItemFiltered.map(({ url, idx }) => {
      const description = descriptions?.[idx]
      return (
        <div className="my-carousel-item" style={{ width }} key={idx}>
          <img
            className={
              description ? "my-carousel-item-img-with-desc" : undefined
            }
            src={url}
            draggable={false}
            alt={`${idx}`}
          />
          {description && (
            <div className="my-carousel-item-desc">{description}</div>
          )}
        </div>
      )
    })
  }, [status, descriptions, imageUrls, slide, moveOption])

  const transformStyle = useMemo(() => {
    const width = `${100 * carouselListDisplay.length}%`

    switch (status) {
      case "dragging":
      case "dragEnding":
        return {
          transform: `translateX(${dragOption.currentTranslateX}px)`,
          width,
        }
      default:
        return { width }
    }
  }, [status, dragOption, carouselListDisplay])

  const transformClass = useMemo(() => {
    const className = "my-carousel-list"
    switch (status) {
      case "dragEnding":
        return className + " transitioning"
      case "moving-left":
        return className + " moving-left"
      case "moving-right":
        return className + " moving-right"
      default:
        return className
    }
  }, [status])

  return (
    <div className="my-carousel-wrapper">
      <div
        className="my-carousel"
        style={{ aspectRatio }}
        ref={carouselRef}
        onMouseDown={(e) => {
          if (imageUrls.length >= 2) {
            click(
              statusRef.current,
              e.clientX,
              e.clientY,
              e.currentTarget.clientWidth,
            )
          }
        }}
        onTouchStart={(e) => {
          if (imageUrls.length >= 2) {
            click(
              statusRef.current,
              e.targetTouches[0].clientX,
              e.targetTouches[0].clientY,
              e.currentTarget.clientWidth,
            )
          }
        }}
      >
        <div className={transformClass} style={transformStyle}>
          {carouselListDisplay}
        </div>
        <div className="my-carousel-control">
          {imageUrls.length >= 2 ? (
            <div
              className="control left"
              onMouseDown={() => {
                if (statusRef.current === "stationary") setClickMove("left")
              }}
              onTouchStart={() => {
                if (statusRef.current === "stationary") setClickMove("left")
              }}
            >
              <i className="bi bi-chevron-left" />
            </div>
          ) : (
            <div />
          )}
          {imageUrls.length >= 2 ? (
            <div
              className="control right"
              onMouseDown={() => {
                if (statusRef.current === "stationary") setClickMove("right")
              }}
              onTouchStart={() => {
                if (statusRef.current === "stationary") setClickMove("right")
              }}
            >
              <i className="bi bi-chevron-right" />
            </div>
          ) : (
            <div />
          )}
        </div>
      </div>
      {imageUrls.length >= 2 && (
        <div className="my-carousel-indicator">
          {imageUrls.map((_, idx) => (
            <button
              key={idx}
              className={`my-indicator${idx === slide ? " active" : ""}`}
              onClick={() =>
                onIndicatorClick(statusRef.current, slideRef.current, idx)
              }
            />
          ))}
        </div>
      )}
    </div>
  )
}
