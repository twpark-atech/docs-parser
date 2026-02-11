import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import HistoryIcon from "@mui/icons-material/History";
import DescriptionIcon from "@mui/icons-material/Description";
import TableChartIcon from "@mui/icons-material/TableChart";

export type NavItem = {
  label: string;
  path: string;
  icon: React.ElementType;
  group?: string;
  primary?: boolean;
};

export const navItems: NavItem[] = [
  { label: "대화하기", path: "/conversations", icon: ChatBubbleOutlineIcon, primary: true },
  { label: "내 채팅 기록", path: "/chatList", icon: HistoryIcon },

  { label: "문서관리", path: "/documentList", icon: DescriptionIcon, group: "지식 자산 관리" },
];
