export interface Camera {
  id: string;
  name: string;
  location: string;
  url: string;
}

export const ALL_CAMERAS: Camera[] = [
  {
    id: "63b65f8dbfd3d90017eaa434",
    name: "Tp. HCM",
    location: "Xô Viết Nghệ Tĩnh - Phan Văn Hân",
    url:
      "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=63b65f8dbfd3d90017eaa434&t=1765179464274",
  },
  {
    id: "56df8108c062921100c143db",
    name: "Tp. HCM",
    location: "Hoàng Minh Giám - Hồng Hà",
    url:
      "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=56df8108c062921100c143db&t=1765520582166",
  },
  {
    id: "5a824ee15058170011f6eab6",
    name: "Tp. HCM",
    location: "Phan Văn Trị - Võ Oanh",
    url:
      "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=5a824ee15058170011f6eab6&t=1765520645741",
  },
  {
    id: "5a824ee15058170011f6eabx6",
    name: "Tp. HCM",
    location: "link video test",
    url:
      "http://14.160.231.254/streams/63f71a4d72b35c0012f8b574/stream/63f71a4d72b35c0012f8b574.m3u8"
  },
  {
    id: "img1",
    name: "Tp. HCM",
    location: "Image",
    url:
      "/pic/img.png"
  },
  {
    id: "6623e5d66f998a001b25235a",
    name: "Tp. HCM",
    location: "Cách Mạng Tháng 8 - Đỗ Thị Lời",
    url:"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e5d66f998a001b25235a&camLocation=C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m%20-%20%C4%90%E1%BB%97%20Th%E1%BB%8B%20L%E1%BB%9Di&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
  },
  {
    id: "5a824ee15058170011f6ea126",
    name: "Tp. HCM",
    location: "HCM - Test",
    url:
      "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=66b1c426779f74001867415e&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20-%20Nguy%E1%BB%85n%20Gia%20Tr%C3%AD&camMode=camera"
  },
];

export const CAMERAS = ALL_CAMERAS.slice(0, 0);// có thể thay đổi tham số truyền vào 0, 0 là ko có cam nào đc chọn sẵn 0, 1 là chọn 1 cam 

